import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch.multiprocessing as mp
from itertools import product
from functools import partial
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util_scripts'))
import selex_meta_utils as meta_utils
import sequence_utils as utils

# Import models from residual trainer to avoid duplication
from residual_trainer import UpdatedCNN, SequenceDataset

class BiasTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()
    
    def train(self, exp_id, control_round, enriched_round, output_dir):
        """Train bias model with hyperparameter optimization"""
        
        os.makedirs(output_dir, exist_ok=True)
        cnn_dir = f"{output_dir}/cnn"
        parameter_dir = f"{output_dir}/parameters"
        os.makedirs(cnn_dir, exist_ok=True)
        os.makedirs(parameter_dir, exist_ok=True)
        
        # Load data
        tf_id = exp_id.split("_")[0]
        source_id = exp_id.split("_")[4]
        
        X, y = self._load_experiment_data(exp_id, source_id, control_round, enriched_round)
        
        # Hyperparameter search
        best_params = self._hyperparameter_search(X, y, tf_id, exp_id, control_round, enriched_round, parameter_dir)
        
        # Train final model
        final_model = self._train_final_model(X, y, best_params, tf_id, exp_id, control_round, enriched_round, cnn_dir)
        
        return final_model, best_params
    
    def _load_experiment_data(self, exp_id, source_id, control_round, enriched_round):
        """Load experiment data"""
        fastqgz_path = f'/usr/project/xtmp/sl548/selex/selex_fastqgz_files/{source_id}'
        rounds, filenames = meta_utils.find_filename_with_exp_id(exp_id)
        
        # Load control sequences
        sequences_control = utils.fastq_to_list(f'{fastqgz_path}/{filenames[rounds.index(control_round)]}')
        sequences_control = [seq for seq in sequences_control if 'N' not in seq]
        if 'bhimsaria2023' in exp_id:
            sequences_control = [seq[:20] for seq in sequences_control]
        
        # Load enriched sequences
        sequences_enriched = utils.fastq_to_list(f'{fastqgz_path}/{filenames[rounds.index(enriched_round)]}')
        sequences_enriched = [seq for seq in sequences_enriched if 'N' not in seq]
        if 'bhimsaria2023' in exp_id:
            sequences_enriched = [seq[:20] for seq in sequences_enriched]
        
        # Combine sequences and create labels
        sequences = sequences_control + sequences_enriched
        X = np.array(sequences)
        y_control = np.zeros(len(sequences_control), dtype=np.float32)
        y_enriched = np.ones(len(sequences_enriched), dtype=np.float32)
        y = np.concatenate([y_control, y_enriched], axis=0)
        
        return X, y
    
    def _hyperparameter_search(self, X, y, tf_id, exp_id, control_round, enriched_round, parameter_dir):
        """Perform hyperparameter search"""
        
        # Hyperparameter combinations
        param_grid = {
            'batch_size': [32, 64, 128],
            'num_filters': [128, 256, 512],
            'kernel_size': [7, 9, 11],
            'n_conv_layers': [1, 2],
            'use_dropout': [False, True]
        }
        
        # Prepare cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # Create parameter combinations
        param_combinations = list(product(
            param_grid['batch_size'],
            param_grid['num_filters'],
            param_grid['kernel_size'],
            param_grid['n_conv_layers'],
            param_grid['use_dropout']
        ))
        
        # Initialize multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # Run parallel hyperparameter search
        num_processes = min(self.num_gpus if self.num_gpus > 0 else 1, len(param_combinations))
        pool = mp.Pool(processes=num_processes)
        
        print(f"Starting hyperparameter search with {num_processes} processes")
        train_func = partial(self._train_single_config, X=X, y=y, kf=kf, num_gpus=self.num_gpus)
        
        try:
            all_results_list = list(tqdm(
                pool.imap(train_func, param_combinations),
                total=len(param_combinations),
                desc="Hyperparameter Search"
            ))
        finally:
            pool.close()
            pool.join()
        
        # Process results
        df_results = []
        best_avg_auroc = 0
        best_params = None
        
        for result in all_results_list:
            if result is None:
                continue
            
            params = result['parameters']
            avg_auroc = result['average_auroc']
            std_auroc = result['std_auroc']
            
            df_results.append({
                'exp_id': exp_id,
                'tf_id': tf_id,
                'round_control': control_round,
                'round_enriched': enriched_round,
                'batch_size': params['batch_size'],
                'num_filters': params['num_filters'],
                'kernel_size': params['kernel_size'],
                'n_conv_layers': params['n_conv_layers'],
                'use_dropout': params['use_dropout'],
                'avg_auroc': avg_auroc,
                'std_auroc': std_auroc
            })
            
            if avg_auroc > best_avg_auroc:
                best_avg_auroc = avg_auroc
                best_params = params
        
        # Save results
        if df_results:
            results_df = pd.DataFrame(df_results)
            os.makedirs(f'{parameter_dir}/{tf_id}', exist_ok=True)
            csv_filename = f'{parameter_dir}/{tf_id}/{exp_id}_{control_round}_{enriched_round}_hyperparameter_results.csv'
            results_df.to_csv(csv_filename, index=False)
            print(f"Best parameters: {best_params}")
            print(f"Best AUROC: {best_avg_auroc:.4f}")
        
        return best_params
    
    def _train_single_config(self, config, X, y, kf, num_gpus):
        """Train single hyperparameter configuration"""
        try:
            batch_size, num_filters, kernel_size, n_conv_layers, use_dropout = config
            try:
                gpu_id = torch.multiprocessing.current_process()._identity[0] % num_gpus
            except:
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}' if num_gpus > 0 else 'cpu')
            
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                try:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    train_dataset = SequenceDataset(X_train, y_train)
                    val_dataset = SequenceDataset(X_val, y_val)
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size)
                    
                    model = UpdatedCNN(num_filters=num_filters, 
                                     kernel_size=kernel_size,
                                     n_conv_layers=n_conv_layers,
                                     use_dropout=use_dropout).to(device)
                    
                    val_auroc = self._train_and_evaluate(model, train_loader, val_loader, device)
                    
                    fold_results.append({
                        'fold': fold + 1,
                        'auroc': float(val_auroc)
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        return None
                    raise e
            
            if not fold_results:
                return None
            
            avg_auroc = np.mean([r['auroc'] for r in fold_results])
            std_auroc = np.std([r['auroc'] for r in fold_results])
            
            return {
                'parameters': {
                    'batch_size': batch_size,
                    'num_filters': num_filters,
                    'kernel_size': kernel_size,
                    'n_conv_layers': n_conv_layers,
                    'use_dropout': use_dropout
                },
                'fold_results': fold_results,
                'average_auroc': float(avg_auroc),
                'std_auroc': float(std_auroc)
            }
            
        except Exception as e:
            return None
    
    def _train_and_evaluate(self, model, train_loader, val_loader, device, num_epochs=10, patience=3):
        """Train and evaluate model"""
        # Calculate class weights
        y_train = torch.cat([y for _, y in train_loader])
        pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-4, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        
        best_val_auroc = 0
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Training
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    logits = model(X_batch).squeeze(-1)
                    
                    preds_prob = torch.sigmoid(logits)
                    val_preds.extend(preds_prob.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())
            
            val_auroc = roc_auc_score(val_labels, val_preds)
            scheduler.step(val_auroc)
            
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        return best_val_auroc
    
    def _train_final_model(self, X, y, best_params, tf_id, exp_id, control_round, enriched_round, cnn_dir):
        """Train final model with best parameters"""
        dataset_all = SequenceDataset(X, y)
        loader_all = DataLoader(dataset_all, batch_size=best_params['batch_size'], shuffle=True)
        
        final_model = UpdatedCNN(num_filters=best_params['num_filters'], 
                              kernel_size=best_params['kernel_size'],
                              n_conv_layers=best_params['n_conv_layers'],
                              use_dropout=best_params['use_dropout']).to(self.device)
        
        # Calculate class weights
        labels = []
        for _, y_batch in loader_all:
            labels.append(y_batch)
        y_all = torch.cat(labels)
        pos_weight = torch.tensor([(1 - y_all.mean()) / y_all.mean()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.Adam(final_model.parameters(), lr=1e-3, eps=1e-4, amsgrad=True)
        
        # Training loop
        final_model.train()
        num_epochs = 10
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            all_preds = []
            all_labels = []
            
            for X_batch, y_batch in loader_all:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = final_model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * X_batch.size(0)
                
                with torch.no_grad():
                    preds_prob = torch.sigmoid(logits)
                    all_preds.extend(preds_prob.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
            
            epoch_loss = total_loss / len(dataset_all)
            epoch_auroc = roc_auc_score(all_labels, all_preds)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}, AUROC: {epoch_auroc:.4f}")
        
        # Save model
        os.makedirs(f'{cnn_dir}/{tf_id}', exist_ok=True)
        model_path = f'{cnn_dir}/{tf_id}/{exp_id}_{control_round}_{enriched_round}_CNN{best_params["kernel_size"]}K{best_params["batch_size"]}B{best_params["num_filters"]}F_L{best_params["n_conv_layers"]}_D{best_params["use_dropout"]}.pt'
        
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'model_config': best_params,
            'final_loss': epoch_loss,
            'final_auroc': epoch_auroc
        }, model_path)
        
        print(f'Saved bias model: {model_path}')
        
        return final_model