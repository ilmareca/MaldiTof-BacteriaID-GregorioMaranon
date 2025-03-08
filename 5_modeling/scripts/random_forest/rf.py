import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import NearMiss, TomekLinks, ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, OneSidedSelection, NeighbourhoodCleaningRule
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
grid_search_results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/random_forest/outputs', 'random_forest_results.csv')
log_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/random_forest/outputs', 'random_forest_log.txt')
os.makedirs(os.path.dirname(grid_search_results_file), exist_ok=True)

print("Loading preprocessed data...")
X = joblib.load(os.path.join(preprocessed_dir, 'X_klebsiella.pkl'))
y = joblib.load(os.path.join(preprocessed_dir, 'y_klebsiella.pkl'))
df_amr = pd.read_csv(os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250304_183343_ciprofloxacina.csv'))

print("Processing data...")
df_amr['extern_id'] = df_amr['extern_id'].astype(str).str.zfill(8)
y_sample = [str(extern_id).zfill(8) for extern_id in y]

df_spectra = pd.DataFrame(X, columns=[f"mz_{i}" for i in range(X.shape[1])])
df_spectra['extern_id'] = y_sample

df_merged = pd.merge(df_spectra, df_amr, on='extern_id')

df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + ['Ciprofloxacina']].dropna()
df_filtered = df_filtered[df_filtered['Ciprofloxacina'].isin(['R', 'S'])]

df_filtered['label'] = df_filtered['Ciprofloxacina'].map({'R': 0, 'S': 1})
X_filtered = df_filtered.drop(columns=['extern_id', 'Ciprofloxacina', 'label']).values
y_filtered = df_filtered['label'].values

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.35, random_state=42, stratify=y_filtered)

def train_and_evaluate(X_train, y_train, X_test, y_test, method):
    print(f"Training model with {method}...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, min_samples_leaf=1, min_samples_split=10, max_features='sqrt', class_weight='balanced')
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['R', 'S'], output_dict=True)
    results = {
        'Method': method,
        'Accuracy': accuracy,
        'Precision_R': report['R']['precision'],
        'Recall_R': report['R']['recall'],
        'F1_R': report['R']['f1-score'],
        'Precision_S': report['S']['precision'],
        'Recall_S': report['S']['recall'],
        'F1_S': report['S']['f1-score']
    }
    print(f"Finished training with {method}.")
    return results

methods = {
    'Original': None,
    'SMOTE': SMOTE(random_state=42),
    'Undersampling': RandomUnderSampler(random_state=42),
    'SMOTEENN': SMOTEENN(random_state=42),
    'Oversampling': RandomOverSampler(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'SVMSmote': SVMSMOTE(random_state=42),
    'BorderlineSmote': BorderlineSMOTE(random_state=42),
    'ClusterCentroids': ClusterCentroids(random_state=42),
    'CondensedNearestNeighbours': CondensedNearestNeighbour(random_state=42),
    'EditedNearestNeighbours': EditedNearestNeighbours(),
    'RepeatedEditedNearestNeightbours': RepeatedEditedNearestNeighbours(),
    'AIIKNN': AllKNN(),
    'InstanceHardnessThreshold': InstanceHardnessThreshold(random_state=42),
    'NearMiss': NearMiss(),
    'NeightbourhoodCleaningRule': NeighbourhoodCleaningRule(),
    'OneSidedSelection': OneSidedSelection(random_state=42),
    'Tomeklinks': TomekLinks()
}

all_results = []

with open(log_file, 'w') as log:
    for method, balancer in methods.items():
        print(f"Processing: {method}")
        log.write(f"\nProcessing: {method}\n")
        
        if balancer:
            X_train_bal, y_train_bal = balancer.fit_resample(X_train, y_train)
        else:
            X_train_bal, y_train_bal = X_train, y_train
        
        results = train_and_evaluate(X_train_bal, y_train_bal, X_test, y_test, method)
        all_results.append(results)
        log.write(str(results) + "\n")

print("Saving results...")
results_df = pd.DataFrame(all_results)
results_df.to_csv(grid_search_results_file, index=False)

print(f"Resultados guardados en {grid_search_results_file}")
