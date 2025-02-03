import os
import shutil
import pandas as pd
import logging
from datetime import datetime


# Function to get the user's choice
def get_user_choice():
    choices = ['A', 'B', 'C', 'D']
    choice = input(f"Select the dataset to process ({'/'.join(choices)}): ").strip().upper()
    while choice not in choices:
        print("Invalid option. Please try again.")
        choice = input(f"Select the dataset to process ({'/'.join(choices)}): ").strip().upper()
    return choice

# Get the user's choice
user_choice = get_user_choice()

# Configure logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/logs/DRIAMS/{user_choice}'
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f'process_{user_choice}_{timestamp}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# Directories
base_dir = f'/export/data_ml4ds/bacteria_id/relevant_datasets/10.5061/dryad.bzkh1899q/DRIAMS_ROOT/DRIAMS-{user_choice}'
csv_dir = os.path.join(base_dir, 'id')
raw_dir = os.path.join(base_dir, 'raw')
output_dir = f'/export/data_ml4ds/bacteria_id/relevant_datasets/DRIAMS_PROCESSED_DATABASE/DRIAMS_{user_choice}'

# Function to process CSV files
def process_csv_files():
    for year in ['2015', '2016', '2017', '2018']:
        csv_file = os.path.join(csv_dir, year, f'{year}_clean.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            for index, row in df.iterrows():
                process_row(row, year)

# Function to process each row of the CSV
def process_row(row, year):
    identifier = row['code']
    print(f'Processing {identifier}...')
    genus_species = row['species'].split(' ')
    if len(genus_species) < 2:
        return
    genus = genus_species[0].capitalize()
    if genus.startswith('Mix!'):
        genus = genus[4:].capitalize()
    species = genus_species[1].split('[')[0].capitalize()
    txt_file = os.path.join(raw_dir, year, f'{identifier}.txt')
    print(f'Copying {txt_file} to {output_dir}/{year}/{genus}/{species}...')
    if os.path.exists(txt_file):
        output_path = os.path.join(output_dir, year, genus, species)
        output_path_TOTAL = os.path.join(output_dir, 'TOTAL', genus, species)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path_TOTAL, exist_ok=True)
        shutil.copy(txt_file, output_path)
        shutil.copy(txt_file, output_path_TOTAL)
        logging.info(f'✔️ Copied {txt_file} to {output_path}')
        logging.info(f'✔️ Copied {txt_file} to {output_path_TOTAL}')
    else:
        logging.info(f'❌ File {txt_file} not found')

# Execute the processing
if __name__ == '__main__':
    process_csv_files()
    logging.info('Processing completed.')