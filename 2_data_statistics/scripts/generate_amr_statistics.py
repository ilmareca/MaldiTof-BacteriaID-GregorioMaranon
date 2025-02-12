import pandas as pd
import os

# Define the path to the input CSV file
input_csv_path = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs/result_amr_20250212_175852.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(input_csv_path)

# Remove the word "Interpretación" from the column names
df.columns = [col.replace(' Interpretación', '') for col in df.columns]

# Create an output directory if it doesn't exist
output_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/2_data_statistics/outputs/'
os.makedirs(output_dir, exist_ok=True)

# Initialize a list to store the counts for each antibiotic
antibiotic_counts = []

# Iterate over each antibiotic column (excluding 'extern_id')
for antibiotic in df.columns[1:]:
    # Count the occurrences of 'S', 'R', and 'I'
    counts = df[antibiotic].value_counts().reindex(['S', 'R', 'I'], fill_value=0)
    
    # Append the counts to the list
    antibiotic_counts.append([antibiotic, counts['S'], counts['R'], counts['I']])

# Create a DataFrame for the counts
counts_df = pd.DataFrame(antibiotic_counts, columns=['Antibiotic', 'S', 'R', 'I'])

# Save the counts to a single CSV file
output_csv_path = os.path.join(output_dir, 'antibiotic_resistance_counts.csv')
counts_df.to_csv(output_csv_path, index=False)

print(f"CSV file generated successfully at {output_csv_path}.")