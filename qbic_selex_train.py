#!/usr/bin/env python3

import argparse
import yaml
import os
import sys
import pandas as pd
from datetime import datetime

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from selex_data_extractor import SelexDataExtractor
from residual_trainer import ResidualTrainer
from ols_trainer import OLSTrainer

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    config_file = os.path.join(os.path.dirname(__file__), config_path)
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_file}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file {config_file}: {e}")

def generate_output_name(exp_id=None, control_round=None, enriched_round=None):
    """Generate output filename"""
    if exp_id and control_round is not None and enriched_round is not None:
        return f"{exp_id}_{control_round}_{enriched_round}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sequences_{timestamp}_{control_round}_{enriched_round}"

def run_single_experiment(args, config):
    """Run pipeline for a single experiment"""
    
    # Initialize components
    data_extractor = SelexDataExtractor(config['data']['fastqgz_path'])
    residual_trainer = ResidualTrainer(config)
    ols_trainer = OLSTrainer(config)
    
    # Stage 1: Data extraction (if needed)
    if args.exp_id:
        print(f"Extracting data for {args.exp_id} (rounds {args.control_round} → {args.enriched_round})")
        control_seqs, enriched_seqs = data_extractor.extract_from_exp_id(
            args.exp_id, args.control_round, args.enriched_round
        )
        output_name = generate_output_name(args.exp_id, args.control_round, args.enriched_round)
    elif args.positive_seqs and args.negative_seqs:
        print(f"Loading sequences from files")
        control_seqs, enriched_seqs = data_extractor.load_sequences_from_files(
            args.positive_seqs, args.negative_seqs
        )
        output_name = generate_output_name(None, args.control_round or 0, args.enriched_round or 1)
    else:
        raise ValueError("Must provide either --exp_id or both --positive_seqs and --negative_seqs")
    
    # Skip stages if specified
    if args.stage and args.stage == 'extract':
        print("Data extraction completed")
        return
    
    # Stage 2: Residual model training
    if not args.stage or args.stage in ['residual', 'all'] or 'residual' in (args.stages or []):
        print("Training residual model...")
        
        bias_model_path = args.bias_model or config['bias_models']['default']
        
        corrected_df = residual_trainer.train(
            control_seqs, enriched_seqs, bias_model_path, output_name,
            num=args.num or config['residual_training']['max_sequences'],
            seed=args.seed or config['residual_training']['seed'],
            save_uncorrected=args.save_uncorrected or config['output']['save_uncorrected'],
            save_model=args.save_models or config['output']['save_models']
        )
        
        if args.stage and args.stage == 'residual':
            print("Residual model training completed")
            return
    
    # Stage 3: OLS training
    if not args.stage or args.stage in ['ols', 'all'] or 'ols' in (args.stages or []):
        print("Training OLS model...")
        
        # Use the alpha-dependent QBiC score selected by ResidualTrainer.
        # For alpha >= 0 this is f_RES; for alpha < 0 this is f_SELEX.
        predictions_csv = f"residual_model_output/corrected/{output_name}.csv"
        
        if not os.path.exists(predictions_csv):
            raise FileNotFoundError(f"Selected QBiC predictions not found: {predictions_csv}. Make sure residual stage completed successfully.")
        
        ols_trainer.train(
            predictions_csv, output_name,
            kmer_size=args.kmer or config['ols_training']['kmer_size'],
            num=args.num,
            seed=args.seed or config['ols_training']['seed'],
            mode=args.mode or config['ols_training']['mode']
        )
        
        print("OLS model training completed")

def run_batch_experiments(batch_file, args, config):
    """Run pipeline for batch of experiments"""
    
    if not os.path.exists(batch_file):
        raise FileNotFoundError(f"Batch file not found: {batch_file}")
    
    # Read batch file
    batch_df = pd.read_csv(batch_file)
    
    # Required columns
    if 'exp_id' not in batch_df.columns:
        raise ValueError("Batch file must contain 'exp_id' column")
    
    # Optional columns with defaults
    if 'control_round' not in batch_df.columns:
        batch_df['control_round'] = args.control_round or 0
    if 'enriched_round' not in batch_df.columns:
        batch_df['enriched_round'] = args.enriched_round or 2
    
    failed_experiments = []
    
    print(f"Processing {len(batch_df)} experiments from batch file...")
    
    for idx, row in batch_df.iterrows():
        exp_id = row['exp_id']
        control_round = row.get('control_round', args.control_round or 0)
        enriched_round = row.get('enriched_round', args.enriched_round or 2)
        
        # Override args with batch-specific values
        batch_args = argparse.Namespace(**vars(args))
        batch_args.exp_id = exp_id
        batch_args.control_round = int(control_round)
        batch_args.enriched_round = int(enriched_round)
        
        # Use batch-specific values if provided
        for col in ['seed', 'kmer_size', 'bias_model']:
            if col in batch_df.columns and pd.notna(row[col]):
                attr_name = col.replace('_size', '').replace('kmer_size', 'kmer')
                if col == 'kmer_size':
                    setattr(batch_args, 'kmer', int(row[col]))
                elif col == 'seed':
                    setattr(batch_args, 'seed', int(row[col]))
                else:
                    setattr(batch_args, attr_name, row[col])
        
        try:
            print(f"\n[{idx+1}/{len(batch_df)}] Processing {exp_id} ({control_round}→{enriched_round})")
            run_single_experiment(batch_args, config)
            print(f"✓ Completed {exp_id}")
            
        except Exception as e:
            import traceback
            error_msg = str(e).replace(',', ';')  # Replace commas to avoid CSV parsing issues
            print(f"✗ Failed {exp_id}: {e}")
            if hasattr(e, '__traceback__'):
                print("Error traceback:")
                traceback.print_exc()
            failed_experiments.append(f"{exp_id},{control_round},{enriched_round},{error_msg}")
    
    # Log failed experiments
    if failed_experiments:
        os.makedirs('logs', exist_ok=True)
        with open('logs/failed_experiments.txt', 'w') as f:
            f.write("exp_id,control_round,enriched_round,error\n")
            for failure in failed_experiments:
                f.write(f"{failure}\n")
        print(f"\n{len(failed_experiments)} experiments failed. See logs/failed_experiments.txt")
    else:
        print(f"\nAll {len(batch_df)} experiments completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="QBiC SELEX Training Pipeline")
    
    # Input options
    parser.add_argument("--exp_id", type=str, help="Experiment ID")
    parser.add_argument("--control_round", type=int, help="Control round number")
    parser.add_argument("--enriched_round", type=int, help="Enriched round number")
    parser.add_argument("--positive_seqs", type=str, help="Positive sequences file")
    parser.add_argument("--negative_seqs", type=str, help="Negative sequences file")
    
    # Batch processing
    parser.add_argument("--batch", type=str, help="Batch file with experiment list")
    
    # Pipeline control
    parser.add_argument("--stage", type=str, choices=['extract', 'residual', 'ols', 'all'], 
                       help="Run specific stage only")
    parser.add_argument("--stages", type=str, help="Comma-separated list of stages to run")
    
    # Model options
    parser.add_argument("--bias_model", type=str, help="Path to bias model")
    parser.add_argument("--kmer", type=int, help="K-mer size for OLS training")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], 
                       help="Output mode: 1=weights only, 2=covariance only, 3=both")
    
    # Training parameters
    parser.add_argument("--num", type=int, help="Maximum number of sequences")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # Output options
    parser.add_argument("--save_uncorrected", action='store_true', help="Save the alternate CNN-derived predictions")
    parser.add_argument("--save_models", action='store_true', help="Save trained models")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Parse stages if provided
    if args.stages:
        args.stages = args.stages.split(',')
    
    # Load configuration
    config = load_config(args.config)
    
    # Run pipeline
    if args.batch:
        run_batch_experiments(args.batch, args, config)
    else:
        run_single_experiment(args, config)

if __name__ == "__main__":
    main()
