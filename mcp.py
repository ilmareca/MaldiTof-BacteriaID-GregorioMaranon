import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Definir rutas de archivos
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250304_183343_ciprofloxacina.csv')

# Cargar los datos preprocesados
X = joblib.load(X_path)
y = joblib.load(y_path)

# Cargar el archivo CSV con información de resistencia antibiótica
df_amr = pd.read_csv(csv_path)

# Asegurar que extern_id tenga 8 dígitos
df_amr['extern_id'] = df_amr['extern_id'].astype(str).str.zfill(8)
y_sample = [str(extern_id).zfill(8) for extern_id in y]

# Crear un DataFrame para los espectros
df_spectra = pd.DataFrame(X, columns=[f"mz_{i}" for i in range(X.shape[1])])
df_spectra['extern_id'] = y_sample

# Unir los datos en extern_id
df_merged = pd.merge(df_spectra, df_amr, on='extern_id')

# Filtrar columnas relevantes
antibiotic = 'Ciprofloxacina'
df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S'])]

# Mapear etiquetas a valores numéricos
labels = df_filtered[antibiotic].map({'R': 0, 'S': 1}).values
X_filtered = df_filtered.drop(columns=['extern_id', antibiotic]).values

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_filtered, labels, test_size=0.35, random_state=42)

# Entrenar modelo de Regresión Logística con los hiperparámetros especificados
best_model = LogisticRegression(C=1000, max_iter=100, penalty='l1', solver='liblinear', random_state=42)
best_model.fit(X_train, y_train)

# Hacer predicciones y evaluar
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['R', 'S'], output_dict=True)

print(f"Precisión del modelo: {accuracy:.4f}")

# Obtener probabilidades de predicción
y_proba = best_model.predict_proba(X_test)

# Calcular la confianza como la probabilidad máxima de la clase predicha
confidence_scores = y_proba.max(axis=1)

# Crear la gráfica de distribución de probabilidades máximas de clase (MCP) y guardarla
plt.figure(figsize=(10, 6))
sns.histplot(confidence_scores, bins=30, kde=True, color='blue', alpha=0.6)
plt.axvline(x=0.75, color='red', linestyle='--', label="Umbral de rechazo (0.75)")
plt.title("Distribución de la probabilidad máxima de clase (MCP)")
plt.xlabel("Máxima probabilidad de clase (MCP)")
plt.ylabel("Frecuencia")
plt.legend()

# Guardar la imagen en lugar de mostrarla
image_path = "mcp_distribution.png"
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Gráfica guardada en {image_path}")

# Guardar resultados en CSV
results = {
    'Accuracy': accuracy,
    'Best_Params': str({'C': 1000, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}),
    'Precision_R': report['R']['precision'],
    'Recall_R': report['R']['recall'],
    'F1_R': report['R']['f1-score'],
    'Precision_S': report['S']['precision'],
    'Recall_S': report['S']['recall'],
    'F1_S': report['S']['f1-score']
}

results_df = pd.DataFrame([results])
results_df.to_csv('logistic_regression_results.csv', index=False)

print("Resultados guardados en 'logistic_regression_results.csv'")
