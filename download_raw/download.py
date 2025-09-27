#!/usr/bin/env python3

# This script is used to download complete raw HT-SELEX data in bulk for QBiC-SELEX training from ENA database and Zenodo.
# Required dependencies:
# 1. sratoolkit.3.2.1-ubuntu64 
# 2. download_list.txt (each row of this file is input for this script, separated by commas, with source, filename_ftp, submitted_ftp)
# 3. sbatch_download.sh (this script is used to submit jobs to the cluster)
# 4. estimated storage space: 310G 


import sys
import os
import subprocess
import argparse
from pathlib import Path


sratoolkit_path = "/usr/project/xtmp/sl548/NCBI/sratoolkit.3.2.1-ubuntu64/bin"

def download_wget(source_id, filename_ftp, submitted_ftp):
    subprocess.run(["wget", submitted_ftp, "-P", source_id])

def download_sratoolkit(source_id, filename_ftp):
    clean_filename = filename_ftp.split(".")[0]
    subprocess.run([f"{sratoolkit_path}/prefetch", clean_filename]) 
    subprocess.run([f"{sratoolkit_path}/fastq-dump", "--gzip", f"{clean_filename}"])
    subprocess.run(["mv", f"{filename_ftp}", source_id])
    subprocess.run(["rm", "-r", f"{clean_filename}"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='a single row in download_list.txt, separated by commas, with source, filename_ftp, submitted_ftp')
    args = parser.parse_args()
    
    input = args.input
    source_id = input.split(",")[0].replace(" ","")
    filename_ftp = input.split(",")[1].replace(" ","")
    submitted_ftp = input.split(",")[2].replace(" ","")

    os.makedirs(source_id, exist_ok=True)
    if Path(f"{source_id}/{filename_ftp}").exists():
        print(f"File {filename_ftp} already exists in {source_id}")
        exit(0)

    if submitted_ftp.startswith("ftp."):
        # Try to download the file up to 3 times if not present
        max_attempts = 3
        attempt = 0
        file_downloaded = False
        while attempt < max_attempts:
            if Path(f"{source_id}/{filename_ftp}").exists():
                file_downloaded = True
                break
            download_wget(source_id, filename_ftp, submitted_ftp)
            attempt += 1

        if not file_downloaded:
            print(f"Failed to download {filename_ftp} in {source_id} after {max_attempts} attempts.")
            with open("failed_to_download.txt", "a") as f:
                f.write(f"{source_id},{filename_ftp},{submitted_ftp}\n")
            raise FileNotFoundError(f"File {filename_ftp} not found in {source_id}")
        
    elif submitted_ftp.startswith("prefetch"):
        max_attempts = 3
        attempt = 0
        file_downloaded = False
        while attempt < max_attempts:
            if Path(f"{source_id}/{filename_ftp}").exists():
                file_downloaded = True
                break
            download_sratoolkit(source_id, filename_ftp)
            attempt += 1

        if not file_downloaded:
            print(f"Failed to download {filename_ftp} in {source_id} after {max_attempts} attempts.")
            with open("failed_to_download.txt", "a") as f:
                f.write(f"{source_id},{filename_ftp},{submitted_ftp}\n")
            raise FileNotFoundError(f"File {filename_ftp} not found in {source_id}")
    
    elif 'zenodo' in submitted_ftp:
        # run wget to download the file

        max_attempts = 1
        attempt = 0
        file_downloaded = False
        while attempt < max_attempts:
            if Path(f"{source_id}/{filename_ftp}").exists():
                file_downloaded = True
                break
            subprocess.run(["wget", submitted_ftp])
            subprocess.run(['tar', '-xvf', "bzip_data_20bp.tar.gz?download=1"])
            subprocess.run(['rm', 'bzip_data_20bp.tar.gz?download=1'])
            subprocess.run(['mv', '20bp/*', source_id])
            subprocess.run(['rm', '-r', '20bp'])
            attempt += 1

        if not Path(f"{source_id}/{filename_ftp}").exists():
            raise FileNotFoundError(f"File {filename_ftp} not found in {source_id}")
            with open("failed_to_download.txt", "a") as f:
                f.write(f"{source_id},{filename_ftp},{submitted_ftp}\n")

    else:
        # skip
        raise ValueError(f"Unsupported FTP: {submitted_ftp}")
        with open("failed_to_download.txt", "a") as f:
            f.write(f"{source_id},{filename_ftp},{submitted_ftp}\n")

