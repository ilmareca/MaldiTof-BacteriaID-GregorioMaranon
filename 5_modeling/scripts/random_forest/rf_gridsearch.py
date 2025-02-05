import os
import joblib
import pandas as pd
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
grid_search_results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/random_forest/outputs', 'grid_search_rf_results.csv')
os.makedirs(os.path.dirname(grid_search_results_file), exist_ok=True)
X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250203_175832_AmpSulbactam.csv')

# Log file paths
hyperparameter_log_file = 'hyperparameter_log.txt'
model_report_log_file = 'model_report_log.txt'

# Function to write logs
def log_hyperparameters(hyperparams):
    with open(hyperparameter_log_file, 'a') as f:
        f.write(f"{hyperparams}\n")

def log_model_report(report, accuracy):
    with open(model_report_log_file, 'a') as f:
        f.write(f"\nAccuracy: {accuracy:.3f}\n")
        f.write("Classification Report:\n")
        f.write(f"{report}\n")

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

# Feature selection function based on correlation
def select_features_by_correlation(df, labels_col, threshold=0.2):
    correlations = []
    features = df.drop(columns=['extern_id', labels_col]).columns
    for feature in features:
        corr = abs(df[feature].corr(df[labels_col].apply(lambda x: 1 if x == 'R' else 0)))
        if corr >= threshold:
            correlations.append((feature, corr))
    
    selected_features = [feature for feature, _ in sorted(correlations, key=lambda x: x[1], reverse=True)]
    return selected_features

# Antibiotic column
antibiotic = 'Amp_Sulbactam_Interpretación'

# UMAP hyperparameter grids
n_neighbors_list = [10, 30, 50]
min_dist_list = [0.1, 0.3, 0.5]
metric_list = ['euclidean', 'manhattan']
threshold_list = [0.1, 0.2, 0.3]

# Random Forest hyperparameter grids
n_estimators_list = [50, 100, 200]
max_depth_list = [10, 20, None]
min_samples_split_list = [2, 5, 10]

# Test size hyperparameter
test_size_list = [0.2, 0.3, 0.4]  # Values to explore for test size

# Store results
grid_search_results = []

# Grid search over UMAP, Random Forest, and test size
for n_neighbors in n_neighbors_list:
    for min_dist in min_dist_list:
        for metric in metric_list:
            for threshold in threshold_list:
                
                # Filter and prepare data
                df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
                df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S', 'I'])]

                # Feature selection based on correlation
                selected_features = select_features_by_correlation(df_filtered, antibiotic, threshold)
                if len(selected_features) < 2:
                    continue  # Skip configurations with insufficient features

                X_filtered = df_filtered[selected_features].values
                labels = df_filtered[antibiotic].map({'R': 0, 'S': 1, 'I': 2}).values  # Map labels to numbers

                # Apply UMAP
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
                X_umap = reducer.fit_transform(X_filtered)

                # Compute silhouette score
                if len(set(labels)) > 1:  # Silhouette score needs at least 2 classes
                    silhouette = silhouette_score(X_umap, labels)
                else:
                    silhouette = float('nan')

                # Grid search over Random Forest and test size
                for test_size in test_size_list:
                    for n_estimators in n_estimators_list:
                        for max_depth in max_depth_list:
                            for min_samples_split in min_samples_split_list:
                                # Log hyperparameters
                                hyperparam_config = {
                                    'n_neighbors': n_neighbors,
                                    'min_dist': min_dist,
                                    'metric': metric,
                                    'threshold': threshold,
                                    'test_size': test_size,
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split
                                }
                                log_hyperparameters(hyperparam_config)

                                # Split data into training and test sets
                                X_train, X_test, y_train, y_test = train_test_split(X_umap, labels, test_size=test_size, random_state=42)

                                # Train Random Forest model
                                rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                                rf_model.fit(X_train, y_train)

                                # Make predictions
                                y_pred = rf_model.predict(X_test)

                                # Evaluate the model
                                accuracy = accuracy_score(y_test, y_pred)
                                report = classification_report(y_test, y_pred, target_names=['R', 'S', 'I'])

                                # Log model report
                                log_model_report(report, accuracy)

                                # Store results
                                report_dict = classification_report(y_test, y_pred, output_dict=True)
                                grid_search_results.append({
                                    'n_neighbors': n_neighbors,
                                    'min_dist': min_dist,
                                    'metric': metric,
                                    'threshold': threshold,
                                    'test_size': test_size,
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'accuracy': accuracy,
                                    'silhouette_score': silhouette,
                                    'precision_R': report_dict['0']['precision'],
                                    'recall_R': report_dict['0']['recall'],
                                    'f1_R': report_dict['0']['f1-score'],
                                    'precision_S': report_dict['1']['precision'],
                                    'recall_S': report_dict['1']['recall'],
                                    'f1_S': report_dict['1']['f1-score'],
                                    'precision_I': report_dict['2']['precision'],
                                    'recall_I': report_dict['2']['recall'],
                                    'f1_I': report_dict['2']['f1-score']
                                })

# Save results to CSV
results_df = pd.DataFrame(grid_search_results)
results_df.to_csv(grid_search_results_file, index=False)

print(f"Grid Search results saved to {grid_search_results_file}")
print(f"Hyperparameter configurations logged in {hyperparameter_log_file}")
print(f"Model reports logged in {model_report_log_file}")
