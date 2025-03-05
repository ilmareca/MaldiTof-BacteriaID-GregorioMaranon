import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Definir rutas
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/gradient_boosting/outputs', 'gradient_boosting_results.csv')
logs_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/gradient_boosting/outputs', 'gradient_boosting_logs.txt')
os.makedirs(os.path.dirname(results_file), exist_ok=True)

# Cargar datos preprocesados
X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250304_183343_ciprofloxacina.csv')

X = joblib.load(X_path)
y = joblib.load(y_path)

# Cargar datos de resistencia
df_amr = pd.read_csv(csv_path)
df_amr['extern_id'] = df_amr['extern_id'].astype(str).str.zfill(8)
y_sample = [str(extern_id).zfill(8) for extern_id in y]

# Crear DataFrame y unir datos
df_spectra = pd.DataFrame(X, columns=[f"mz_{i}" for i in range(X.shape[1])])
df_spectra['extern_id'] = y_sample
df_merged = pd.merge(df_spectra, df_amr, on='extern_id')

# Filtrar por Ciprofloxacina
antibiotic = 'Ciprofloxacina'
df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S'])]

# Mapear etiquetas
labels = df_filtered[antibiotic].map({'R': 0, 'S': 1}).values
X_filtered = df_filtered.drop(columns=['extern_id', antibiotic]).values

# Split inicial sin balanceo
X_train, X_test, y_train, y_test = train_test_split(X_filtered, labels, test_size=0.35, random_state=42)

# GridSearch para Gradient Boosting
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
gb = GradientBoostingClassifier()
grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

# Guardar mejores parámetros
with open(logs_file, 'a') as log:
    log.write(f"Mejores parámetros: {best_params}\n")

# Función para evaluar el modelo con distintas técnicas de balanceo
def evaluate_model(X, y, method_name, resampler=None):
    results_list = []

    for seed in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed)

        # Aplicar resampling si se proporciona un método
        if resampler:
            X_train, y_train = resampler.fit_resample(X_train, y_train)

        # Entrenar modelo con los mejores parámetros
        model = GradientBoostingClassifier(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        precision_r = report['0']['precision']
        recall_r = report['0']['recall']
        f1_r = report['0']['f1-score']
        precision_s = report['1']['precision']
        recall_s = report['1']['recall']
        f1_s = report['1']['f1-score']

        results_list.append([method_name, seed, acc, precision_r, recall_r, f1_r, precision_s, recall_s, f1_s])

    # Convertir a DataFrame
    results_df = pd.DataFrame(results_list, columns=['Method', 'Seed', 'Accuracy', 'Precision_R', 'Recall_R', 'F1_R', 'Precision_S', 'Recall_S', 'F1_S'])

    # Calcular media y desviación
    metrics_summary = results_df.iloc[:, 2:].agg(['mean', 'std']).reset_index()
    metrics_summary.rename(columns={'index': 'Metric'}, inplace=True)

    # Guardar resultados en CSV
    results_df.to_csv(results_file, mode='a', index=False, header=not os.path.exists(results_file))
    metrics_summary.to_csv(results_file, mode='a', index=False, header=False)

    # Guardar resultados en logs
    with open(logs_file, 'a') as log:
        log.write(f"\nResultados con {method_name}:\n")
        log.write(results_df.to_string(index=False) + "\n")
        log.write("\nResumen de métricas (Media y Desviación Estándar):\n")
        log.write(metrics_summary.to_string(index=False) + "\n")

    print(f"Resultados de {method_name} guardados en CSV y logs.")

# **Paso 2: Evaluación sin balanceo**
evaluate_model(X_filtered, labels, "Sin Balanceo")

# **Paso 3: Evaluación con SMOTE**
smote = SMOTE(random_state=42)
evaluate_model(X_filtered, labels, "SMOTE", smote)

# **Paso 4: Evaluación con SMOTEENN**
smoteenn = SMOTEENN(random_state=42)
evaluate_model(X_filtered, labels, "SMOTEENN", smoteenn)

# **Paso 5: Evaluación con Random Oversampling**
oversampler = RandomOverSampler(random_state=42)
evaluate_model(X_filtered, labels, "Oversampling", oversampler)

# **Paso 6: Evaluación con Random Undersampling**
undersampler = RandomUnderSampler(random_state=42)
evaluate_model(X_filtered, labels, "Undersampling", undersampler)

print(f"Todos los resultados han sido guardados en {results_file} y {logs_file}.")
