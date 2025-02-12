import os
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import umap

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
output_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/4_feature_extraction/scripts/umap/figures/raw'
os.makedirs('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/4_feature_extraction/scripts/umap/outputs', exist_ok=True)
X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250212_175852_Imipenem.csv')

# Load the preprocessed data
X = joblib.load(X_path)
y = joblib.load(y_path)

# Normalización usando MinMaxScaler para preservar la forma de los espectros

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

# List of antibiotics to analyze
antibiotics = [col for col in df_amr.columns if 'Interpretación' in col]

# Define colors for labels
color_map = {'R': '#da0d91', 'S': '#07b457'}

# Define the grid of hyperparameters for UMAP and feature selection
n_neighbors_list = [2, 5, 10, 15, 30, 45]  # Different neighborhood sizes
min_dist_list = [0.1, 0.3, 0.5, 0.8]  # Different minimum distances
metric_list = ['euclidean', 'manhattan']  # Different distance metrics
variance_threshold_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]  # Different thresholds for variance selection

# Store results
results = []

# Perform UMAP for each antibiotic and hyperparameter combination
for antibiotic in antibiotics:
    # Filter and prepare the data
    df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
    df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S'])]

    # Extract features and labels
    X_features = df_filtered.drop(columns=['extern_id', antibiotic]).values
    labels = df_filtered[antibiotic].map({'R': 0, 'S': 1}).values  

    # Iterate over different variance thresholds for feature selection
    for threshold in variance_threshold_list:

        # Iterate over different hyperparameter combinations
        for n_neighbors in n_neighbors_list:
            for min_dist in min_dist_list:
                for metric in metric_list:
                    # Perform UMAP with the current hyperparameter combination
                    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
                    X_umap = reducer.fit_transform(X_features)

                    # Create a DataFrame for visualization
                    df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
                    df_umap['label'] = df_filtered[antibiotic].values
                    df_umap['color'] = df_umap['label'].map(color_map)

                    # Plot
                    plt.figure(figsize=(10, 7))
                    plt.scatter(df_umap['UMAP1'], df_umap['UMAP2'], c=df_umap['color'], alpha=0.7, s=15, edgecolor='k', linewidth=0.3)

                    # Titles and labels
                    plt.title(f'UMAP ({antibiotic})\n n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}, variance={threshold}')
                    plt.xlabel('UMAP1')
                    plt.ylabel('UMAP2')

                    # Add legend
                    legend_labels = df_umap['label'].unique()
                    handles = [plt.Line2D([0], [0], marker='o', color=color_map[label], markersize=8) for label in legend_labels]
                    plt.legend(handles, legend_labels, title='Antibiotic Resistance', bbox_to_anchor=(1.05, 1), loc='upper left')

                    # Save the plot with a descriptive filename
                    safe_antibiotic_name = antibiotic.replace('/', '_').replace(' ', '_')
                    file_name = f'umap_{safe_antibiotic_name}_n{n_neighbors}_dist{min_dist}_metric{metric}_var{threshold}.png'
                    os.makedirs(output_dir, exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, file_name))
                    plt.close()  # Close the plot to free memory
                    print(f"Saved: {file_name}")

                    # Store the results
                    results.append([antibiotic, n_neighbors, min_dist, metric, threshold, file_name])


