import os
import joblib
import pandas as pd
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek  # Cambio de SMOTE a SMOTETomek para manejar mejor el desbalance

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
grid_search_results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/random_forest/outputs', 'rf_umap_smote_f1_optimization.csv')
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

# Apply UMAP with finer neighborhood structure
reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, metric='euclidean', random_state=42)
X_umap = reducer.fit_transform(X_filtered)

# Apply SMOTETomek to balance the classes
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_umap, labels)

# Reduce test set size for additional training data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Refine GridSearch parameters with finer increments
param_grid = {
    'n_estimators': [200, 500, 1000, 1200],
    'max_depth': [15, 20, 30, 40],
    'min_samples_split': [2, 4, 6, 10],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Initialize Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Apply GridSearchCV optimizing f1_macro
grid_search = GridSearchCV(
    estimator=rf_model, 
    param_grid=param_grid, 
    cv=3, 
    scoring='f1_macro', 
    n_jobs=-1, 
    verbose=2
)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['R', 'S', 'I'])

# Print results
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)
print(f"Accuracy: {accuracy:.3f}")
print("Classification Report:")
print(report)

# Save the results to a CSV file
results_dict = classification_report(y_test, y_pred, output_dict=True)
results_df = pd.DataFrame([{
    'Accuracy': accuracy,
    'Best_Params': str(grid_search.best_params_),
    'Precision_R': results_dict['0']['precision'],
    'Recall_R': results_dict['0']['recall'],
    'F1_R': results_dict['0']['f1-score'],
    'Precision_S': results_dict['1']['precision'],
    'Recall_S': results_dict['1']['recall'],
    'F1_S': results_dict['1']['f1-score'],
    'Precision_I': results_dict['2']['precision'],
    'Recall_I': results_dict['2']['recall'],
    'F1_I': results_dict['2']['f1-score']
}])

results_df.to_csv(grid_search_results_file, index=False)

print(f"Grid Search results saved to {grid_search_results_file}")
