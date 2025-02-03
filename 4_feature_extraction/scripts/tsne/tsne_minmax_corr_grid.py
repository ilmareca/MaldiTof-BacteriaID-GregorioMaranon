import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
output_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/4_feature_extraction/scripts/tsne/figures/minmax_corr'
score_output_file = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/4_feature_extraction/scripts/tsne/outputs', 'tsne_minmax_corr_grid.csv')
os.makedirs('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/4_feature_extraction/scripts/tsne/outputs', exist_ok=True)
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

# List of antibiotics to analyze
antibiotics = [col for col in df_amr.columns if 'Interpretación' in col]

# Define colors for labels
color_map = {'R': '#da0d91', 'S': '#07b457', 'I': '#f4a742'}

# Define the grid of hyperparameters for t-SNE and feature selection
perplexity_list = [5, 10, 30, 50]  # Different perplexity values
learning_rate_list = [10, 50, 200]  # Different learning rates
threshold_list = [0.05, 0.1, 0.15, 0.2]  # Different thresholds for feature selection

# Perform feature selection based on correlation
def select_features_by_correlation(df, labels_col, threshold=0.1):
    correlations = []
    features = df.drop(columns=['extern_id', labels_col]).columns
    for feature in features:
        corr = abs(df[feature].corr(df[labels_col].apply(lambda x: 1 if x == 'R' else 0)))
        if corr >= threshold:
            correlations.append((feature, corr))
    
    # Ordenar por la magnitud de la correlación y seleccionar las mejores
    selected_features = [feature for feature, _ in sorted(correlations, key=lambda x: x[1], reverse=True)]
    return selected_features

# Store results
results = []

# Perform t-SNE for each antibiotic and hyperparameter combination
for antibiotic in antibiotics:
    # Filter and prepare the data
    df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
    df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S', 'I'])]

    # Iterate over different thresholds for feature selection
    for threshold in threshold_list:
        # Select features based on correlation
        selected_features = select_features_by_correlation(df_filtered, antibiotic, threshold)
        if len(selected_features) < 2:
            print(f"Skipping {antibiotic} with threshold {threshold} due to insufficient features.")
            continue

        X_filtered = df_filtered[selected_features].values
        labels = df_filtered[antibiotic].map({'R': 0, 'S': 1, 'I': 2}).values  # Map to numerical labels for silhouette score

        # Iterate over different hyperparameter combinations
        for perplexity in perplexity_list:
            for learning_rate in learning_rate_list:
                # Perform t-SNE with the current hyperparameter combination
                tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
                X_tsne = tsne.fit_transform(X_filtered)

                # Compute silhouette score
                if len(set(labels)) > 1:  # Silhouette score needs at least 2 classes
                    silhouette = silhouette_score(X_tsne, labels)
                else:
                    silhouette = float('nan')  # Assign NaN if only one class exists

                # Create a DataFrame for visualization
                df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
                df_tsne['label'] = df_filtered[antibiotic].values
                df_tsne['color'] = df_tsne['label'].map(color_map)

                # Plot
                plt.figure(figsize=(10, 7))
                plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], c=df_tsne['color'], alpha=0.7, s=15, edgecolor='k', linewidth=0.3)

                # Titles and labels
                plt.title(f't-SNE ({antibiotic})\nScore: {silhouette:.3f}, perplexity={perplexity}, learning_rate={learning_rate}, threshold={threshold}')
                plt.xlabel('TSNE1')
                plt.ylabel('TSNE2')

                # Add legend
                legend_labels = df_tsne['label'].unique()
                handles = [plt.Line2D([0], [0], marker='o', color=color_map[label], markersize=8) for label in legend_labels]
                plt.legend(handles, legend_labels, title='Antibiotic Resistance', bbox_to_anchor=(1.05, 1), loc='upper left')

                # Save the plot with a descriptive filename
                safe_antibiotic_name = antibiotic.replace('/', '_').replace(' ', '_')
                file_name = f'tsne_{safe_antibiotic_name}_perp{perplexity}_lr{learning_rate}_thresh{threshold}.png'
                os.makedirs(output_dir, exist_ok=True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, file_name))
                plt.close()  # Close the plot to free memory
                print(f"Saved: {file_name}")

                # Store the results
                results.append([antibiotic, perplexity, learning_rate, threshold, silhouette, file_name])

# Save the results to a CSV file
results_df = pd.DataFrame(results, columns=['Antibiotic', 'perplexity', 'learning_rate', 'threshold', 'silhouette_score', 'file_name'])
results_df.to_csv(score_output_file, index=False)
print(f"Scores saved to {score_output_file}")
