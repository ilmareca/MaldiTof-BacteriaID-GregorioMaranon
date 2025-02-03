import os
import joblib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

# Define paths
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/preprocessing/amr'
output_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/preprocessing/2d-visual/tsne/minmax_thres'

pkl_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/amr'
X_path = os.path.join(pkl_dir, 'X_BINNED.pkl')
y_path = os.path.join(pkl_dir, 'y_BINNED.pkl')
csv_path = os.path.join(preprocessed_dir, 'result_amr_20250128_180939_aztreonam.csv')

# Load the preprocessed data
X = joblib.load(X_path)
y = joblib.load(y_path)

# Normalización usando MinMaxScaler para preservar la forma de los espectros
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Selección de características: elimina picos con baja varianza
selector = VarianceThreshold(threshold=0.01)  # Umbral ajustable
X_selected = selector.fit_transform(X_scaled)
print(f"Dimensión reducida: {X_scaled.shape} -> {X_selected.shape}")

# Load the CSV file with antibiotic resistance information
df_amr = pd.read_csv(csv_path)

# Ensure extern_id has 8 digits
df_amr['extern_id'] = df_amr['extern_id'].astype(str).str.zfill(8)
y_sample = [str(extern_id).zfill(8) for extern_id in y]

# Create a DataFrame for the spectra
df_spectra = pd.DataFrame({'extern_id': y_sample, 'spectrum': list(X_selected)})

# Merge the data on extern_id
df_merged = pd.merge(df_spectra, df_amr, on='extern_id')

# List of antibiotics to analyze
antibiotics = [col for col in df_amr.columns if 'Interpretación' in col]

# Define colors for labels
color_map = {'R': '#da0d91', 'S': '#07b457', 'I': '#f4a742'}

# Perform t-SNE for each antibiotic
for antibiotic in antibiotics:
    # Filter the data for the current antibiotic
    df_filtered = df_merged[['spectrum', antibiotic]].dropna()

    # Only include rows where the antibiotic classification is 'R', 'S', o 'I'
    df_filtered = df_filtered[df_filtered[antibiotic].isin(['R', 'S', 'I'])]

    # Extract the spectra and labels
    X_filtered = np.array(list(df_filtered['spectrum']))
    labels = df_filtered[antibiotic].values

    # Adjust perplexity based on the number of samples
    n_samples = len(X_filtered)
    if n_samples <= 1:
        print(f"Skipping {antibiotic} due to insufficient samples.")
        continue
    perplexity = min(10, n_samples // 3)  # Perplexity más baja ajustada

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=5000)  # Iteraciones aumentadas
    X_tsne = tsne.fit_transform(X_filtered)

    # Create a DataFrame for visualization
    df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    df_tsne['label'] = labels

    # Mapear los colores correctamente
    df_tsne['color'] = df_tsne['label'].map(color_map)

    # Verificar que no haya valores NaN en los colores
    if df_tsne['color'].isna().any():
        raise ValueError("Se encontraron etiquetas no mapeadas a colores.")

    # Visualización del scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], c=df_tsne['color'], alpha=0.7, s=30)

    # Título y etiquetas
    plt.title(f't-SNE Visualization of Spectra with {antibiotic}')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')

    # Añadir la leyenda
    legend_labels = df_tsne['label'].unique()
    handles = [plt.Line2D([0], [0], marker='o', color=color_map[label], markersize=10) for label in legend_labels]
    plt.legend(handles, legend_labels, title='Antibiotic Resistance', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Guardar y mostrar la imagen
    plt.tight_layout()
    safe_antibiotic_name = antibiotic.replace('/', '_').replace(' ', '_')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'tsne_visualization_{safe_antibiotic_name}.png'))
    plt.show()
