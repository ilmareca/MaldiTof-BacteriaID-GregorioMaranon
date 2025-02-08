import os
import joblib
import pandas as pd
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE  # SMOTE para manejar clases desbalanceadas

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
grid_search_results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/random_forest/outputs', 'grid_search_rf_with_smote_results.csv')
os.makedirs(os.path.dirname(grid_search_results_file), exist_ok=True)

# Ficheros de logs
hyperparameter_log_file = 'smote_hyperparameter_log.txt'
model_results_log_file = 'smote_model_results_log.txt'

# Funciones para escribir logs
def log_hyperparameters(config):
    with open(hyperparameter_log_file, 'a') as f:
        f.write(f"Hyperparameters: {config}\n")

def log_model_results(config, accuracy, report):
    with open(model_results_log_file, 'a') as f:
        f.write(f"Hyperparameters: {config}\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write("Classification Report:\n")
        f.write(f"{classification_report_to_string(report)}\n")
        f.write("--------------------------------------------------\n")

def classification_report_to_string(report):
    return f"Class 0 (R): Precision={report['0']['precision']:.3f}, Recall={report['0']['recall']:.3f}, F1={report['0']['f1-score']:.3f}\n" \
           f"Class 1 (S): Precision={report['1']['precision']:.3f}, Recall={report['1']['recall']:.3f}, F1={report['1']['f1-score']:.3f}\n" \
           f"Class 2 (I): Precision={report['2']['precision']:.3f}, Recall={report['2']['recall']:.3f}, F1={report['2']['f1-score']:.3f}\n"

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

# Hiperparámetros para el Grid Search
umap_n_neighbors = [10, 30, 50]
umap_min_dist = [0.1, 0.3, 0.5]
umap_metrics = ['euclidean', 'manhattan']

rf_n_estimators = [100, 200]
rf_max_depth = [None, 10, 20]
rf_min_samples_split = [2, 5]

# Almacenar los resultados
grid_search_results = []

# Iterar sobre todas las combinaciones de parámetros
for n_neighbors in umap_n_neighbors:
    for min_dist in umap_min_dist:
        for metric in umap_metrics:
            # Aplicar UMAP
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
            X_umap = reducer.fit_transform(X_filtered)

            # Calcular el número de muestras originales por clase
            original_class_counts = pd.Series(labels).value_counts()

            for smote_ratio in [0.5, 0.7, 1.0]:
                # Calcular la estrategia de muestreo asegurando al menos el número de muestras originales
                sampling_strategy = {}
                for label, count in original_class_counts.items():
                    target_count = int(max(count, count * smote_ratio))  # Asegura que sea al menos igual al original
                    sampling_strategy[label] = target_count

                # Aplicar SMOTE
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_umap, labels)

                # Dividir los datos
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

                for n_estimators in rf_n_estimators:
                    for max_depth in rf_max_depth:
                        for min_samples_split in rf_min_samples_split:
                            # Configuración de hiperparámetros
                            hyperparam_config = {
                                'n_neighbors': n_neighbors,
                                'min_dist': min_dist,
                                'metric': metric,
                                'smote_ratio': smote_ratio,
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split
                            }

                            # Registrar hiperparámetros
                            log_hyperparameters(hyperparam_config)

                            # Entrenar el modelo Random Forest
                            rf_model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=42,
                                class_weight='balanced'
                            )
                            rf_model.fit(X_train, y_train)

                            # Realizar predicciones
                            y_pred = rf_model.predict(X_test)

                            # Evaluar el modelo
                            accuracy = accuracy_score(y_test, y_pred)
                            report = classification_report(y_test, y_pred, output_dict=True)

                            # Registrar resultados del modelo
                            log_model_results(hyperparam_config, accuracy, report)

                            # Almacenar los resultados
                            grid_search_results.append({
                                'n_neighbors': n_neighbors,
                                'min_dist': min_dist,
                                'metric': metric,
                                'smote_ratio': smote_ratio,
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'accuracy': accuracy,
                                'precision_R': report['0']['precision'],
                                'recall_R': report['0']['recall'],
                                'f1_R': report['0']['f1-score'],
                                'precision_S': report['1']['precision'],
                                'recall_S': report['1']['recall'],
                                'f1_S': report['1']['f1-score'],
                                'precision_I': report['2']['precision'],
                                'recall_I': report['2']['recall'],
                                'f1_I': report['2']['f1-score']
                            })

# Guardar los resultados en un archivo CSV
results_df = pd.DataFrame(grid_search_results)
results_df.to_csv(grid_search_results_file, index=False)

print(f"Grid Search results saved to {grid_search_results_file}")
print(f"Hyperparameter configurations logged in {hyperparameter_log_file}")
print(f"Model results logged in {model_results_log_file}")
