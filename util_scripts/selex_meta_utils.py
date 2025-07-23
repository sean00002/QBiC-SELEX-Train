import pandas as pd # type: ignore
import numpy as np # type: ignore

# this script needs to pair with selex_meta_mapped.csv

import os

# Try different possible locations for metadata file
metadata_paths = [
    os.path.join(os.path.dirname(__file__), 'metadata_experiments.csv'),  # util_scripts/metadata_experiments.csv
    './metadata_experiments.csv',  # current directory
    '/Users/shengyuli/Library/CloudStorage/Dropbox/Research/qbic2/utils/metadata_experiments.csv'  # original path
]

exp_meta = None
for path in metadata_paths:
    try:
        exp_meta = pd.read_csv(path)
        break
    except FileNotFoundError:
        continue

if exp_meta is None:
    raise FileNotFoundError("metadata_experiments.csv not found in any expected location")

"""
try:
    model_meta = pd.read_csv('/Users/shengyuli/Library/CloudStorage/Dropbox/Research/qbic2/utils/metadata_models.csv')
except FileNotFoundError:
    model_meta = pd.read_csv('/usr/project/xtmp/sl548/qbic2/utils/metadata_models.csv')
"""

def find_filename_with_exp_id(exp_id, selex_ena_all=exp_meta):
    """
    This function finds the filenames for a given exp_id
    """
    if 'bhimsaria' in exp_id:
        return find_filename_with_exp_id_bhimsaria(exp_id, selex_ena_all)

    exp_data = selex_ena_all[selex_ena_all['exp_id'] == exp_id]
    exp_data = exp_data.sort_values(by = 'round')
    filenames = list(exp_data['filename_ftp'].values)
    rounds = list(exp_data['round'].values)
    
    barcode, source = exp_id.split("_")[2], exp_id.split("_")[-1]
    
    # Use boolean indexing for better performance
    zero_mask = (selex_ena_all['tf'].str.contains('ZERO')) & \
                (selex_ena_all['source'] == source) & \
                (selex_ena_all['barcode'] == barcode)
    
    round_0 = selex_ena_all[zero_mask]['filename_ftp'].values

    if len(round_0) > 0:
        rounds = [0]*len(round_0) + list(rounds)
        filenames = list(round_0) + list(filenames)

    return [int(x) for x in rounds], filenames 

def find_filename_with_pair_id(pair_id):
    """
    This function finds the filenames for a given pair_id   
    """

    if 'bhimsaria' in pair_id:
        rounds, filenames = find_filename_with_exp_id_bhimsaria('_'.join(pair_id.split('_')[:5]))
        control_cycle_num = str(pair_id.split('_')[5])
        enriched_cycle_num = str(pair_id.split('_')[6])
        return [control_cycle_num, enriched_cycle_num], [filenames[rounds.index(control_cycle_num)], filenames[rounds.index(enriched_cycle_num)]]

        
    exp_id = '_'.join(pair_id.split('_')[:5])
    control_cycle_num = int(pair_id.split('_')[5])
    enriched_cycle_num = int(pair_id.split('_')[6])
    rounds, filenames = find_filename_with_exp_id(exp_id)
    
    control_filename = filenames[rounds.index(control_cycle_num)]
    enriched_filename = filenames[rounds.index(enriched_cycle_num)]

    return [int(control_cycle_num), int(enriched_cycle_num)], [control_filename, enriched_filename]

def find_filename_with_exp_id_bhimsaria(exp_id, selex_ena_all=exp_meta):
    source = exp_id.split("_")[-1]
    source_data = selex_ena_all[selex_ena_all['source'] == source]
    
    exp_data = source_data[source_data['exp_id'] == exp_id].sort_values(by = 'round')
    exp_rounds = list(exp_data['round'].values)
    exp_filenames = list(exp_data['filename_ftp'].values)
    
    barcode = exp_id.split("_")[2]
    zero_rounds = source_data[source_data['tf'].str.contains('ZERO') & (source_data['barcode'] == barcode)]['round'].values
    zero_filenames = source_data[source_data['tf'].str.contains('ZERO') & (source_data['barcode'] == barcode)]['filename_ftp'].values

    all_rounds = list(zero_rounds) + list(exp_rounds)
    all_filenames = list(zero_filenames) + list(exp_filenames)

    return all_rounds, all_filenames

# find pair ids for a given exp_id
def output_pair_ids(exp_ids):
    """
    This function finds the pair ids for a given exp_id
    """
    pair_ids = []
    filenames1 = []
    filenames2 = []
    if isinstance(exp_ids, str):
        exp_ids = [exp_ids]
        
    for exp_id in exp_ids:
        rounds, filenames = find_filename_with_exp_id(exp_id)

        rounds = [int(x) for x in rounds]
        
        control_rounds = [x for x in rounds if x <= 2]
        
        for round1 in control_rounds:
            for round2 in rounds:
                if round2 > round1:
                    pair_ids.append(f"{exp_id}_{round1}_{round2}")
                    filenames1.append(filenames[rounds.index(round1)])
                    filenames2.append(filenames[rounds.index(round2)])
    
    return pair_ids, filenames1, filenames2



