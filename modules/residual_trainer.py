import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_filters=100, kernel_size=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size)
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = self.conv1(x) 
        x = F.relu(x)
        x = self.global_max_pool(x)
        x = x.squeeze(-1)
        logits = self.fc(x)
        return logits

class UpdatedCNN(nn.Module):
    def __init__(self, num_filters=128, kernel_size=7, n_conv_layers=2, use_dropout=False, dropout_rate=0.2):
        super(UpdatedCNN, self).__init__()
        
        self.use_dropout = use_dropout
        layers = []
        in_channels = 4
        
        for i in range(n_conv_layers):
            conv = nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2)
            bn = nn.BatchNorm1d(num_filters)
            relu = nn.ReLU()
            layers.extend([conv, bn, relu])
            in_channels = num_filters
        
        self.conv_stack = nn.Sequential(*layers)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        
        self.fc_hidden = nn.Linear(num_filters, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.global_max_pool(x)
        x = x.squeeze(-1)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc_hidden(x))
        logits = self.out(x)
        return logits

class ResidualModel(nn.Module):
    def __init__(self, bias_model, num_filters=128, kernel_size=7):
        super(ResidualModel, self).__init__()
        self.bias_model = bias_model
        self.bias_scale = nn.Parameter(torch.zeros(1))
        self.residual_branch = SimpleCNN(num_filters=num_filters, kernel_size=kernel_size)
    
    def forward(self, x, return_residual=False):
        with torch.no_grad():
            bias_logit = self.bias_model(x)
        
        residual_logit = self.residual_branch(x)
        constrained_scale = torch.tanh(self.bias_scale)
        tf_logit = constrained_scale * bias_logit + residual_logit
        
        if return_residual:
            return tf_logit, residual_logit
        return tf_logit

def one_hot_encode_sequence(seq):
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in base_to_idx:
            arr[base_to_idx[base], i] = 1.0
    return arr

class SequenceDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        self.X = []
        for seq in seqs:
            self.X.append(one_hot_encode_sequence(seq))
        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

class ResidualTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, control_sequences, enriched_sequences, bias_model_path, output_name, 
              num=1800000, seed=42, save_uncorrected=False, save_model=False):
        """Train residual model and generate predictions"""
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Prepare data
        if not control_sequences:
            raise ValueError("No control sequences provided")
        if not enriched_sequences:
            raise ValueError("No enriched sequences provided")
        
        print(f"Training with {len(control_sequences)} control and {len(enriched_sequences)} enriched sequences")
        
        sequences = control_sequences + enriched_sequences
        X = np.array(sequences)
        y_control = np.zeros(len(control_sequences), dtype=np.float32)
        y_enriched = np.ones(len(enriched_sequences), dtype=np.float32)
        y = np.concatenate([y_control, y_enriched], axis=0)
        
        # Load bias model
        bias_model = self._load_bias_model(bias_model_path)
        
        # Train residual model
        residual_model, optimizer, epoch_loss, epoch_auroc, bias_scale_value, dataset_all = self._train_residual_model(
            bias_model, X, y
        )
        
        # Generate predictions
        corrected_df, uncorrected_df = self._generate_predictions(
            residual_model, dataset_all, sequences, num, seed, bias_scale_value
        )
        
        # Save outputs
        self._save_outputs(corrected_df, uncorrected_df, residual_model, optimizer, 
                          epoch_loss, epoch_auroc, bias_scale_value, output_name,
                          save_uncorrected, save_model)
        
        return corrected_df
    
    def _load_bias_model(self, bias_model_path):
        """Load and prepare bias model"""
        if not os.path.exists(bias_model_path):
            raise FileNotFoundError(f"Bias model not found: {bias_model_path}")
        
        print(f"Loading bias model from {bias_model_path}")
        bias_model_checkpoint = torch.load(bias_model_path, map_location=self.device)
        bias_model = UpdatedCNN(
            num_filters=bias_model_checkpoint['model_config']['num_filters'],
            kernel_size=bias_model_checkpoint['model_config']['kernel_size'],
            n_conv_layers=bias_model_checkpoint['model_config']['n_conv_layers'],
            use_dropout=bias_model_checkpoint['model_config']['use_dropout']
        )
        bias_model.load_state_dict(bias_model_checkpoint['model_state_dict'])
        bias_model.eval()
        bias_model = bias_model.to(self.device)
        
        # Freeze parameters
        for param in bias_model.parameters():
            param.requires_grad = False
        
        return bias_model
    
    def _train_residual_model(self, bias_model, X, y):
        """Train the residual model"""
        dataset_all = SequenceDataset(X, y)
        loader_all = DataLoader(dataset_all, batch_size=64, shuffle=True)
        
        residual_model = ResidualModel(bias_model, num_filters=128, kernel_size=9).to(self.device)
        
        # Calculate class weights
        labels = []
        for _, y_batch in loader_all:
            labels.append(y_batch)
        y_all = torch.cat(labels)
        pos_weight = torch.tensor([(1 - y_all.mean()) / y_all.mean()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.Adam(residual_model.parameters(), lr=1e-3, eps=1e-4, amsgrad=True)
        
        # Training loop
        residual_model.train()
        num_epochs = 7
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            all_preds = []
            all_labels = []
            
            for X_batch, y_batch in loader_all:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = residual_model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(residual_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * X_batch.size(0)
                
                with torch.no_grad():
                    preds_prob = torch.sigmoid(logits)
                    all_preds.extend(preds_prob.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
            
            epoch_loss = total_loss / len(dataset_all)
            epoch_auroc = roc_auc_score(all_labels, all_preds)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}, AUROC: {epoch_auroc:.4f}")
        
        bias_scale_value = torch.tanh(residual_model.bias_scale).item()
        
        return residual_model, optimizer, epoch_loss, epoch_auroc, bias_scale_value, dataset_all
    
    def _generate_predictions(self, residual_model, dataset_all, sequences, num, seed, bias_scale_value):
        """Generate prediction dataframes"""
        residual_model.eval()
        
        loader_ordered = DataLoader(dataset_all, batch_size=64, shuffle=False)
        all_logits = []
        all_residuals = []
        
        with torch.no_grad():
            for X_batch, _ in loader_ordered:
                X_batch = X_batch.to(self.device)
                
                batch_logit, batch_residuals = residual_model(X_batch, return_residual=True)
                batch_logit = batch_logit.squeeze(-1)
                batch_residuals = batch_residuals.squeeze(-1)
                all_logits.append(batch_logit.cpu().numpy())
                all_residuals.append(batch_residuals.cpu().numpy())
        
        all_logits = np.concatenate(all_logits, axis=0)
        all_residuals = np.concatenate(all_residuals, axis=0)
        
        # Create dataframes
        combined_df = pd.DataFrame({'sequence': sequences, 'score': all_logits})
        combined_df = combined_df.drop_duplicates(subset=['sequence'], keep='first')
        if combined_df.shape[0] > num:
            combined_df = combined_df.sample(n=num, random_state=seed)
        
        residual_df = pd.DataFrame({'sequence': sequences, 'score': all_residuals})
        residual_df = residual_df.drop_duplicates(subset=['sequence'], keep='first')
        if residual_df.shape[0] > num:
            residual_df = residual_df.sample(n=num, random_state=seed)
        
        if bias_scale_value >= 0:
            return combined_df, residual_df
        else:
            return residual_df, combined_df
    
    def _save_outputs(self, corrected_df, uncorrected_df, residual_model, optimizer,
                     epoch_loss, epoch_auroc, bias_scale_value, output_name,
                     save_uncorrected, save_model):
        """Save model outputs"""
        
        # Always save corrected predictions
        os.makedirs('residual_model_output/corrected', exist_ok=True)
        corrected_df.to_csv(f'residual_model_output/corrected/{output_name}.csv', index=False)
        
        # Optionally save uncorrected predictions
        if save_uncorrected:
            os.makedirs('residual_model_output/uncorrected', exist_ok=True)
            uncorrected_df.to_csv(f'residual_model_output/uncorrected/{output_name}.csv', index=False)
        
        # Optionally save model
        if save_model:
            os.makedirs('models/residual', exist_ok=True)
            model_path = f'models/residual/{output_name}.pt'
            
            torch.save({
                'model_state_dict': residual_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 7,
                'loss': epoch_loss,
                'auroc': epoch_auroc,
                'bias_scale': bias_scale_value,
                'model_config': {
                    'num_filters': 128,
                    'kernel_size': 9,
                    'batch_size': 64,
                }
            }, model_path)