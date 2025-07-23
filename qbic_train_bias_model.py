#!/usr/bin/env python3

import argparse
import yaml
import os
import sys

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from bias_trainer import BiasTrainer

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    config_file = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="QBiC Bias Model Training")
    
    parser.add_argument("-i", "--experiment_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("-e", "--enriched_round", type=int, required=True, help="Enriched round")
    parser.add_argument("-c", "--control_round", type=int, required=True, help="Control round")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize and run bias trainer
    trainer = BiasTrainer(config)
    
    print(f"Training bias model for {args.experiment_id}")
    print(f"Control round: {args.control_round}, Enriched round: {args.enriched_round}")
    print(f"Output directory: {args.output}")
    
    model, best_params = trainer.train(
        args.experiment_id, 
        args.control_round, 
        args.enriched_round, 
        args.output
    )
    
    print("Bias model training completed!")
    print(f"Best parameters: {best_params}")

if __name__ == "__main__":
    main()