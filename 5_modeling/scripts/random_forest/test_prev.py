import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import umap
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
grid_search_results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/random_forest/outputs', 'grid_search_rf_results.csv')
os.makedirs('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/random_forest/outputs', exist_ok=True)
X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250203_175832_AmpSulbactam.csv')

# Load the preprocessed data
X = joblib.load(X_path)
y = joblib.load(y_path)

# Normalización usando MinMaxScaler para preservar la forma de los espectros
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
    
    # Ordenar por la magnitud de la correlación y seleccionar las mejores
    selected_features = [feature for feature, _ in sorted(correlations, key=lambda x: x[1], reverse=True)]
    return selected_features

# Hiperparámetros fijos para UMAP
antibiotic = 'Amp_Sulbactam_Interpretación'
n_neighbors = 30
min_dist = 0.1
metric = 'euclidean'
threshold = 0.2

# Filtrar y preparar los datos
df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S', 'I'])]

# Selección de características basadas en correlación
selected_features = select_features_by_correlation(df_filtered, antibiotic, threshold)
if len(selected_features) < 2:
    raise ValueError(f"Insufficient features selected with threshold {threshold}. Please adjust the threshold.")

X_filtered = df_filtered[selected_features].values
labels = df_filtered[antibiotic].map({'R': 0, 'S': 1, 'I': 2}).values  # Mapear etiquetas a números

# Aplicar UMAP
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
X_umap = reducer.fit_transform(X_filtered)

# Silhouette score
if len(set(labels)) > 1:  # Silhouette score necesita al menos 2 clases
    silhouette = silhouette_score(X_umap, labels)
else:
    silhouette = float('nan')

# Grid de hiperparámetros de Random Forest
n_estimators_list = [50, 100, 200]
max_depth_list = [10, 20, None]
min_samples_split_list = [2, 5, 10]

# Almacenar resultados
grid_search_results = []

# Bucle sobre el grid de Random Forest
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X_umap, labels, test_size=0.5, random_state=42)

            # Entrenar el modelo de Random Forest
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            rf_model.fit(X_train, y_train)

            # Predicciones
            y_pred = rf_model.predict(X_test)

            # Evaluar el modelo
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)  # Obtenemos un diccionario con las métricas

            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {accuracy:.3f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['R', 'S', 'I']))



            # Guardar resultados
            grid_search_results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'accuracy': accuracy,
                'silhouette_score': silhouette,
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

# Convertir resultados a DataFrame y guardarlos en CSV
results_df = pd.DataFrame(grid_search_results)
results_df.to_csv(grid_search_results_file, index=False)

print(f"Grid Search results saved to {grid_search_results_file}")
