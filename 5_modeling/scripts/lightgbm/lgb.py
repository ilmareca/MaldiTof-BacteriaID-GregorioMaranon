import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/lightgbm/outputs', 'lightgbm_results.csv')
log_file_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/lightgbm/outputs', 'lightgbm_training.log')
os.makedirs(os.path.dirname(results_file), exist_ok=True)
X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250304_183343_ciprofloxacina.csv')

# Function to write logs
def write_log(message):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')
        print(message)

# Load the preprocessed data
X = joblib.load(X_path)
y = joblib.load(y_path)

# Load the CSV file with antibiotic resistance information
df_amr = pd.read_csv(csv_path)

# Ensure extern_id has 8 digits
df_amr['extern_id'] = df_amr['extern_id'].astype(str).str.zfill(8)
y_sample = [str(extern_id).zfill(8) for extern_id in y]

# Create a DataFrame for the spectra
df_spectra = pd.DataFrame(X, columns=[f"mz_{i}" for i in range(X.shape[1])])
df_spectra['extern_id'] = y_sample

# Merge the data on extern_id
df_merged = pd.merge(df_spectra, df_amr, on='extern_id')

# Antibiotic column
antibiotic = 'Ciprofloxacina'

# Filter and prepare data
df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S'])]

# Map labels to numerical values
labels = df_filtered[antibiotic].map({'R': 0, 'S': 1}).values
X_filtered = df_filtered.drop(columns=['extern_id', antibiotic]).values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, labels, test_size=0.35, random_state=42)
write_log("Data loaded and split into training and test sets.")

# Initialize LightGBM model
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=500, learning_rate=0.1, max_depth=-1, num_leaves=31)

# Train the model
write_log("Training LightGBM model...")
lgb_model.fit(X_train, y_train)
write_log("Model training completed.")

# Make predictions
y_pred = lgb_model.predict(X_test)
write_log("Predictions completed.")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['R', 'S'], output_dict=True)

# Store results
results = {
    'Accuracy': accuracy,
    'Precision_R': report['R']['precision'],
    'Recall_R': report['R']['recall'],
    'F1_R': report['R']['f1-score'],
    'Precision_S': report['S']['precision'],
    'Recall_S': report['S']['recall'],
    'F1_S': report['S']['f1-score']
}

# Save results to CSV
results_df = pd.DataFrame([results])
results_df.to_csv(results_file, index=False)

write_log(f"Results saved to {results_file}")
