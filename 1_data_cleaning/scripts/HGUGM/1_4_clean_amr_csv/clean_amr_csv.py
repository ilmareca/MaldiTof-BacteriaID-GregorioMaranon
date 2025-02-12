import os
import pandas as pd
import logging
from datetime import datetime

# Define paths
base_dir = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon'
data_cleaner_dir = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/klebsiellaPneumoniae'
csv_path = os.path.join(base_dir, 'AMR_HUGM_2018-2023.csv')  # Replace with your actual CSV file name
output_dir  = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs'
os.makedirs(output_dir, exist_ok=True)

# Create logs directory if it doesn't exist
logs_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/logs/HGUGM/amr-csv-cleaner'
os.makedirs(logs_dir, exist_ok=True)

# Configure loggers
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(f'{logs_dir}/error_log_{timestamp}.log')
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)

found_logger = logging.getLogger('found_logger')
found_handler = logging.FileHandler(f'{logs_dir}/found_log_{timestamp}.log')
found_logger.addHandler(found_handler)
found_logger.setLevel(logging.INFO)

not_found_logger = logging.getLogger('not_found_logger')
not_found_handler = logging.FileHandler(f'{logs_dir}/not_found_log_{timestamp}.log')
not_found_logger.addHandler(not_found_handler)
not_found_logger.setLevel(logging.INFO)

# Read the CSV file with the correct encoding and handle BOM
try:
    with open(csv_path, 'r', encoding='utf-8-sig') as file:
        df = pd.read_csv(file, sep=',')  # Ensure the correct delimiter is used
except UnicodeDecodeError as e:
    error_logger.error(f"Error reading CSV file: {e}")
    raise

# Print column names for debugging
print("Column names in the CSV file:")
print(df.columns)

# Verify column names
expected_columns = ['Muestra'] + [col for col in df.columns if 'Interpretación' in col]
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    error_logger.error(f"Missing columns in CSV file: {missing_columns}")
    raise KeyError(f"Missing columns in CSV file: {missing_columns}")

# Initialize the result DataFrame
columns = ['extern_id'] + [col for col in df.columns if 'Interpretación' in col]
result_df = pd.DataFrame(columns=columns)

# Mapping dictionary for interpretation values
interpretation_mapping = {
    'S': 'S',
    'R': 'R',
    'I': 'R',  # Map intermediate to resistant as DRIAMS does
    'R*': 'R',
    'ESBL': '',
    'EBL?': ''
}

# Process each row in the CSV
for index, row in df.iterrows():
    try:
        extern_id = str(int(float(row['Muestra']))).zfill(8)  # Convert to float, then to int to remove decimal, then to str and zfill
    except ValueError as e:
        error_logger.error(f"Invalid Muestra value at index {index}: {row['Muestra']}")
        continue
    
    extern_id_dir = os.path.join(data_cleaner_dir, extern_id)
    
    if os.path.exists(extern_id_dir):
        found_logger.info(f"Found extern_id: {extern_id}")
        result_row = {'extern_id': extern_id}
        has_antibiotic_data = False
        for col in columns[1:]:
            value = row[col]
            if pd.notna(value) and value != '':
                has_antibiotic_data = True
            result_row[col] = interpretation_mapping.get(value, value)  # Map interpretation values
        
        if has_antibiotic_data:
            result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        not_found_logger.info(f"Not found extern_id: {extern_id}")

# Save the result DataFrame to a CSV file
result_csv_path = os.path.join(output_dir, f'result_amr_{timestamp}.csv')
result_df.to_csv(result_csv_path, index=False)

print(f"Processing complete. Results saved to {result_csv_path}")