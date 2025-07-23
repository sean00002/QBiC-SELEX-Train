import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util_scripts'))
import selex_meta_utils as meta_utils
import sequence_utils as utils

class SelexDataExtractor:
    def __init__(self, fastqgz_path='/usr/project/xtmp/sl548/selex/selex_fastqgz_files/'):
        self.fastqgz_path = fastqgz_path
    
    def extract_from_exp_id(self, exp_id, control_round, enriched_round):
        """Extract sequences from experiment ID and round numbers"""
        source_id = exp_id.split("_")[4]
        
        if 'bhimsaria' not in exp_id:
            return self._extract_normal_experiment(exp_id, source_id, control_round, enriched_round)
        else:
            return self._extract_bhimsaria_experiment(exp_id, control_round, enriched_round)
    
    def _extract_normal_experiment(self, exp_id, source_id, control_round, enriched_round):
        """Extract sequences from normal experiments"""
        rounds, filenames = meta_utils.find_filename_with_exp_id(exp_id)
        
        sequences_control = self._load_sequences_for_round(
            source_id, filenames, rounds, control_round
        )
        sequences_enriched = self._load_sequences_for_round(
            source_id, filenames, rounds, enriched_round
        )
        
        return sequences_control, sequences_enriched
    
    def _extract_bhimsaria_experiment(self, exp_id, control_round, enriched_round):
        """Extract sequences from Bhimsaria experiments with special handling"""
        rounds, filenames = meta_utils.find_filename_with_exp_id_bhimsaria(exp_id)
        
        # Load control sequences
        filename_control = filenames[rounds.index(str(control_round))]
        file_path = f'{self.fastqgz_path}/bhimsaria2023/{filename_control}'
        sequences_control = utils.fastq_to_list(file_path)
        sequences_control = [seq for seq in sequences_control if 'N' not in seq]
        sequences_control = [x[:20] for x in sequences_control]  # Trim to first 20 chars
        
        # Load enriched sequences
        filename_enriched = filenames[rounds.index(str(enriched_round))]
        file_path = f'{self.fastqgz_path}/bhimsaria2023/{filename_enriched}'
        sequences_enriched = utils.fastq_to_list(file_path)
        sequences_enriched = [seq for seq in sequences_enriched if 'N' not in seq]
        sequences_enriched = [x[:20] for x in sequences_enriched]  # Trim to first 20 chars
        
        return sequences_control, sequences_enriched
    
    def _load_sequences_for_round(self, source_id, filenames, rounds, round_num):
        """Load sequences for a specific round"""
        try:
            filename = filenames[rounds.index(round_num)]
        except ValueError:
            raise ValueError(f"Round {round_num} not found in available rounds: {rounds}")
        
        file_path = f'{self.fastqgz_path}/{source_id}/{filename}'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if filename.endswith('.txt'):  # for rm2017 experiments 
            sequences = []
            with open(file_path, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
        else:
            sequences = utils.fastq_to_list(file_path)
        
        # Remove sequences with N
        sequences = [seq for seq in sequences if 'N' not in seq]
        
        if not sequences:
            raise ValueError(f"No valid sequences found in {file_path}")
        
        return sequences
    
    def load_sequences_from_files(self, positive_file, negative_file):
        """Load sequences directly from files"""
        positive_seqs = self._load_sequence_file(positive_file)
        negative_seqs = self._load_sequence_file(negative_file)
        return negative_seqs, positive_seqs  # control, enriched order
    
    def _load_sequence_file(self, filepath):
        """Load sequences from various file formats"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Sequence file not found: {filepath}")
        
        if filepath.endswith('.txt'):
            with open(filepath, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
        elif filepath.endswith('.fasta') or filepath.endswith('.fa'):
            from Bio import SeqIO
            with open(filepath, 'r') as f:
                sequences = [str(record.seq) for record in SeqIO.parse(f, 'fasta')]
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            if 'sequence' not in df.columns:
                raise ValueError(f"CSV file {filepath} must contain 'sequence' column")
            sequences = df['sequence'].tolist()
        else:
            raise ValueError(f"Unsupported file format: {filepath}. Supported formats: .txt, .fasta, .fa, .csv")
        
        if not sequences:
            raise ValueError(f"No sequences found in {filepath}")
        
        # Remove sequences containing 'N'
        original_count = len(sequences)
        sequences = [seq for seq in sequences if 'N' not in seq]
        if len(sequences) < original_count:
            print(f"Warning: Removed {original_count - len(sequences)} sequences containing 'N' nucleotides")
        
        return sequences