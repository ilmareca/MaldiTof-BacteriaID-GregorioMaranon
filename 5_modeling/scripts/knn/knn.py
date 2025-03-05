import os
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Definir rutas
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/knn/outputs', 'knn_results.csv')
os.makedirs(os.path.dirname(results_file), exist_ok=True)

X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250304_183343_ciprofloxacina.csv')

# Cargar datos preprocesados
X = joblib.load(X_path)
y = joblib.load(y_path)

# Cargar archivo CSV con información de resistencia antibiótica
df_amr = pd.read_csv(csv_path)

# Asegurar que 'extern_id' tenga 8 dígitos
df_amr['extern_id'] = df_amr['extern_id'].astype(str).str.zfill(8)
y_sample = [str(extern_id).zfill(8) for extern_id in y]

# Crear DataFrame de espectros
df_spectra = pd.DataFrame(X, columns=[f"mz_{i}" for i in range(X.shape[1])])
df_spectra['extern_id'] = y_sample

# Unir datos en 'extern_id'
df_merged = pd.merge(df_spectra, df_amr, on='extern_id')

# Selección de antibiótico
antibiotic = 'Ciprofloxacina'

# Filtrar y preparar datos
df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S'])]

# Mapear etiquetas a valores numéricos
labels = df_filtered[antibiotic].map({'R': 0, 'S': 1}).values
X_filtered = df_filtered.drop(columns=['extern_id', antibiotic]).values

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_filtered, labels, test_size=0.35, random_state=42)

# Definir y entrenar modelo KNN con los mejores parámetros
knn_model = KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='distance')
knn_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = knn_model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['R', 'S'], output_dict=True)

# Guardar resultados
results = {
    'Accuracy': accuracy,
    'Metric': 'manhattan',
    'n_neighbors': 3,
    'Weights': 'distance',
    'Precision_R': report['R']['precision'],
    'Recall_R': report['R']['recall'],
    'F1_R': report['R']['f1-score'],
    'Precision_S': report['S']['precision'],
    'Recall_S': report['S']['recall'],
    'F1_S': report['S']['f1-score']
}

# Guardar resultados en CSV
results_df = pd.DataFrame([results])
results_df.to_csv(results_file, index=False)

print(f"Resultados de KNN guardados en {results_file}")
