import os
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.manifold import TSNE

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
tsne_output_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/4_feature_extraction/scripts/tsne/figures/raw'
os.makedirs('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/4_feature_extraction/scripts/tsne/outputs', exist_ok=True)
X_path = os.path.join(preprocessed_dir, 'X_klebsiella.pkl')
y_path = os.path.join(preprocessed_dir, 'y_klebsiella.pkl')
csv_path = os.path.join('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/scripts/HGUGM/1_4_clean_amr_csv/outputs', 'result_amr_20250212_175852_Imipenem.csv')

# Load the preprocessed data
X = joblib.load(X_path)
y = joblib.load(y_path)

# Load the CSV file with antibiotic resistance information
df_amr = pd.read_csv(csv_path)

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

# Define the grid of hyperparameters for t-SNE and feature selection
perplexity_list = [5, 10, 30, 50]  # Different perplexity values
learning_rate_list = [10, 50, 100, 200]  # Different learning rates

# Store results
results = []

# Perform t-SNE for each antibiotic and hyperparameter combination
for antibiotic in antibiotics:
    df_filtered = df_merged[['extern_id'] + [col for col in df_merged.columns if col.startswith('mz_')] + [antibiotic]].dropna()
    df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S'])]
    
    X_features = df_filtered.drop(columns=['extern_id', antibiotic]).values
    labels = df_filtered[antibiotic].map({'R': 0, 'S': 1}).values  

    
    for perplexity in perplexity_list:
        for learning_rate in learning_rate_list:
            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
            X_tsne = tsne.fit_transform(X_features)
            
            df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
            df_tsne['label'] = df_filtered[antibiotic].values
            df_tsne['color'] = df_tsne['label'].map(color_map)
            
            plt.figure(figsize=(10, 7))
            plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], c=df_tsne['color'], alpha=0.7, s=15, edgecolor='k', linewidth=0.3)
            
            plt.title(f't-SNE ({antibiotic})\n Perplexity={perplexity}, Learning rate={learning_rate}')
            plt.xlabel('TSNE1')
            plt.ylabel('TSNE2')
            
            legend_labels = df_tsne['label'].unique()
            handles = [plt.Line2D([0], [0], marker='o', color=color_map[label], markersize=8) for label in legend_labels]
            plt.legend(handles, legend_labels, title='Antibiotic Resistance', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            safe_antibiotic_name = antibiotic.replace('/', '_').replace(' ', '_')
            file_name = f'tsne_{safe_antibiotic_name}_perp{perplexity}_lr{learning_rate}.png'
            os.makedirs(tsne_output_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(tsne_output_dir, file_name))
            plt.close()
            print(f"Saved: {file_name}")
            
            results.append([antibiotic, perplexity, learning_rate, file_name])
