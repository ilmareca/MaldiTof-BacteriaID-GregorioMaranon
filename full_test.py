import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed


# Definir rutas
dir_output = 'logs'
os.makedirs(dir_output, exist_ok=True)
log_file = os.path.join(dir_output, 'training_results.log')

def log_results(model_name, accuracy, report):
    with open(log_file, 'a') as f:
        f.write(f"\n{model_name} Results:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision R: {report['R']['precision']:.4f}\n")
        f.write(f"Recall R: {report['R']['recall']:.4f}\n")
        f.write(f"F1 R: {report['R']['f1-score']:.4f}\n")
        f.write(f"Precision S: {report['S']['precision']:.4f}\n")
        f.write(f"Recall S: {report['S']['recall']:.4f}\n")
        f.write(f"F1 S: {report['S']['f1-score']:.4f}\n")
        f.write("--------------------------------------------\n")

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
grid_search_results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon', 'full_test.csv')
os.makedirs(os.path.dirname(grid_search_results_file), exist_ok=True)
X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250304_183343_ciprofloxacina.csv')

# Load the preprocessed data
X = joblib.load(X_path)
y = joblib.load(y_path)

# Load the CSV file with antibiotic resistance information
df_amr = pd.read_csv(csv_path)

# Ensure extern_id has 8 digits
df_amr['extern_id'] = df_amr['extern_id'].astype(str).str.zfill(8)
y_sample = [str(extern_id).zfill(8) for extern_id in y]

df_spectra = pd.DataFrame(X, columns=[f"mz_{i}" for i in range(X.shape[1])])
df_spectra['extern_id'] = y_sample
df_merged = pd.merge(df_spectra, df_amr, on='extern_id')

df_filtered = df_merged.dropna()
df_filtered = df_filtered[df_filtered['Ciprofloxacina'].isin(['R', 'S'])]

y_labels = df_filtered['Ciprofloxacina'].map({'R': 0, 'S': 1}).values
X_features = df_filtered.drop(columns=['extern_id', 'Ciprofloxacina']).values

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.4, random_state=42, stratify=y_labels)

# Aplicar RFECV con Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfe = RFECV(estimator=rf, cv=5, step=0.1, n_jobs=-1, min_features_to_select=96)
rfe.fit(X_train, y_train)

X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Modelos a entrenar
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "GaussianNB": GaussianNB(),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
}

# Entrenar y evaluar cada modelo
for model_name, model in models.items():
    model.fit(X_train_rfe, y_train)
    y_pred = model.predict(X_test_rfe)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['R', 'S'], output_dict=True)
    log_results(model_name, accuracy, report)

# DNN
set_random_seed(42)
dnn = Sequential([
    Input(shape=(X_train_rfe.shape[1],)),
    Dense(10, activation='relu', name='Hidden'),
    Dropout(0.15, name='Dropout'),
    Dense(1, activation='sigmoid', name='Output')
])
dnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
dnn.fit(X_train_rfe, y_train, epochs=20, batch_size=32, verbose=0, validation_data=(X_test_rfe, y_test))

y_pred_dnn = (dnn.predict(X_test_rfe) > 0.5).astype(int).flatten()
accuracy_dnn = accuracy_score(y_test, y_pred_dnn)
report_dnn = classification_report(y_test, y_pred_dnn, target_names=['R', 'S'], output_dict=True)
log_results("DNN", accuracy_dnn, report_dnn)

print("Entrenamiento y evaluación completados. Revisa los logs en 'logs/training_results.log'")
