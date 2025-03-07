import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
grid_search_results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/lightgbm/outputs', 'grid_search_lightgbm_results.csv')
os.makedirs(os.path.dirname(grid_search_results_file), exist_ok=True)
X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250203_175832_AmpSulbactam.csv')

# Load the preprocessed data
X = joblib.load(X_path)
y = joblib.load(y_path)

# Normalize using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load the CSV file with antibiotic resistance information
df_amr = pd.read_csv(csv_path)

# Ensure extern_id has 8 digits
df_amr['extern_id'] = df_amr['extern_id'].astype(str).str.zfill(8)
y_sample = [str(extern_id).zfill(8) for extern_id in y]

# Create a DataFrame for the spectra
df_spectra = pd.DataFrame(X_scaled, columns=[f"mz_{i}" for i in range(X_scaled.shape[1])])
df_spectra['extern_id'] = y_sample

# Merge the data on extern_id
df_merged = pd.merge(df_spectra, df_amr, on='extern_id')

# Antibiotic column
antibiotic = 'Amp_Sulbactam_Interpretación'

# Filter and prepare data
df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S', 'I'])]

# Map labels to numerical values
labels = df_filtered[antibiotic].map({'R': 0, 'S': 1, 'I': 2}).values
X_filtered = df_filtered.drop(columns=['extern_id', antibiotic]).values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, labels, test_size=0.35, random_state=42)

# Define LightGBM and GridSearch parameters
param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 500]
}

# Initialize LightGBM classifier
lightgbm_model = lgb.LGBMClassifier(random_state=42)

# Apply GridSearchCV optimizing f1_macro
grid_search = GridSearchCV(
    estimator=lightgbm_model, 
    param_grid=param_grid, 
    cv=3, 
    scoring='f1_macro', 
    n_jobs=2, 
    verbose=2
)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['R', 'S', 'I'], output_dict=True)

# Store results
results = {
    'Accuracy': accuracy,
    'Best_Params': str(grid_search.best_params_),
    'Precision_R': report['R']['precision'],
    'Recall_R': report['R']['recall'],
    'F1_R': report['R']['f1-score'],
    'Precision_S': report['S']['precision'],
    'Recall_S': report['S']['recall'],
    'F1_S': report['S']['f1-score'],
    'Precision_I': report['I']['precision'],
    'Recall_I': report['I']['recall'],
    'F1_I': report['I']['f1-score']
}

# Save results to CSV
results_df = pd.DataFrame([results])
results_df.to_csv(grid_search_results_file, index=False)

print(f"Grid Search results saved to {grid_search_results_file}")
