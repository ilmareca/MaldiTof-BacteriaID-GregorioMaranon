import os
import joblib
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
grid_search_results_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/5_modeling/scripts/pike/outputs', 'raw_pike_gridsearch.csv')
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

# Create a DataFrame for the spectra
df_spectra = pd.DataFrame(X, columns=[f"mz_{i}" for i in range(X.shape[1])])
df_spectra['extern_id'] = y_sample

# Merge the data on extern_id
df_merged = pd.merge(df_spectra, df_amr, on='extern_id')

# Antibiotic column
antibiotic = 'Ciprofloxacina'

# Filter and prepare data
df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S'])]

# Map labels to numerical values
labels = df_filtered[antibiotic].map({'R': 0, 'S': 1}).values
X_filtered = df_filtered.drop(columns=['extern_id', antibiotic]).values

# Define PIKE kernel class
class PIKE(Kernel):
    def __init__(self, t=1, n_jobs=10):
        self.t = t
        self.n_jobs = n_jobs

    def __call__(self, X_mz, X_i, Y_mz=None, Y_i=None, distance_TH=1e-6):
        if Y_mz is None and Y_i is None:
            Y_mz = X_mz
            Y_i = X_i

        K = np.zeros((X_mz.shape[0], Y_mz.shape[0]))
        positions_x = X_mz[0, :].reshape(-1, 1)
        positions_y = Y_mz[0, :].reshape(-1, 1)
        distances = pairwise_distances(positions_x, positions_y, metric='sqeuclidean')
        distances = np.exp(-distances / (4 * self.t))
        d = np.where(distances[0] < distance_TH)[0][0]

        def compute_partial_sum(i, x, X_i, Y_i, distances, d):
            intensities_y = Y_i.T[:(i+d), :]
            di = distances[i, :(i+d)].reshape(-1, 1)
            prod = intensities_y * di
            x = np.broadcast_to(x, (np.minimum(i+d, X_i.shape[1]), X_i.shape[0])).T
            return np.matmul(x, prod)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_partial_sum)(i, x, X_i, Y_i, distances, d) for i, x in enumerate(X_i.T)
        )

        K = np.sum(results, axis=0) / (4 * self.t * np.pi)
        return K

# Initialize PIKE kernel
pike_kernel = PIKE(t=1)

# Define GridSearch parameters
param_grid = {'max_iter_predict': [50, 100, 200], 'n_restarts_optimizer': [0, 1, 2]}

# Initialize Gaussian Process Classifier (GPC) with PIKE kernel
gpc = GaussianProcessClassifier(kernel=pike_kernel)
grid_search = GridSearchCV(estimator=gpc, param_grid=param_grid, cv=LeaveOneOut(), n_jobs=-1)

# Perform data augmentation and train the model
accuracies = []
for _ in range(100):
    spec1C_mz, spec1C_intensity, labels_1C = X_filtered, X_filtered.T, labels
    K_train = pike_kernel(X_i=spec1C_intensity, X_mz=spec1C_mz)
    grid_search.fit(K_train, labels_1C)
    accuracies.append(grid_search.best_score_)

# Compute mean and standard deviation of accuracy
mean_accuracy = 100 * np.mean(accuracies)
std_accuracy = 100 * np.sqrt(np.var(accuracies))

# Save results to CSV
results = {
    'Mean_Accuracy': mean_accuracy,
    'Std_Accuracy': std_accuracy,
    'Best_Params': str(grid_search.best_params_)
}
results_df = pd.DataFrame([results])
results_df.to_csv(grid_search_results_file, index=False)

print(f"Grid Search results saved to {grid_search_results_file}")
print("The accuracy is %2.2f ± %2.2f" % (mean_accuracy, std_accuracy))
