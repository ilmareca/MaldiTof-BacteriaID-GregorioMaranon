import os
import numpy as np
import matplotlib.pyplot as plt
from spectrum import SpectrumObject, VarStabilizer, Smoother, BaselineCorrecter, Normalizer, Trimmer, Binner
from sklearn.manifold import TSNE
from scipy import interpolate

# Define the base directory and species names
base_dir_template = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/{year}/matched_bacteria'
species_names = [
    'Streptococcus_Agalactiae', 'Citrobacter_Freundii', 'Candida_Albicans',
    'Proteus_Mirabilis', 'Pseudomonas_Aeruginosa', 'Staphylococcus_Epidermidis',
    'Klebsiella_Pneumoniae', 'Staphylococcus_Aureus', 'Enterococcus_Faecalis',
    'Escherichia_Coli'
]
years = [2018, 2019, 2020, 2021, 2022, 2023]

# Function to load spectra from multiple years
def load_spectra_multiple_years(species_name, base_dir_template, years):
    acqu_files = []
    fid_files = []
    
    for year in years:
        base_dir = base_dir_template.format(year=year)
        species_dir = os.path.join(base_dir, species_name.replace("_", "/"))
        
        # Traverse all subdirectories for acqu and fid files
        for root, dirs, files in os.walk(species_dir):
            for file in files:
                if file.endswith("acqu"):
                    acqu_files.append(os.path.join(root, file))
                elif file.endswith("fid"):
                    fid_files.append(os.path.join(root, file))

    return acqu_files, fid_files

# Function to preprocess spectra with variance stabilization, smoothing, baseline correction, normalization, and trimming
def preprocess_spectrum(acqu_file, fid_file, target_length=1000):
    spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)
    
    # Apply each preprocessing step
    spectrum = variance_stabilization(spectrum)
    spectrum = smoothing(spectrum)
    spectrum = baseline_correction(spectrum)
    spectrum = normalization(spectrum)
    spectrum = trimming(spectrum)
    spectrum = binning(spectrum)
    
    # Interpolate to the target length
    mz_new, intensity_new = interpolate_spectrum(spectrum, target_length)
    return intensity_new

# Preprocessing steps
def variance_stabilization(spectrum):
    stabilizer = VarStabilizer(method="sqrt")
    return stabilizer(spectrum)

def smoothing(spectrum):
    smoother = Smoother(halfwindow=10, polyorder=3)
    return smoother(spectrum)

def baseline_correction(spectrum):
    baseline_correcter = BaselineCorrecter(method="SNIP", snip_n_iter=10)
    return baseline_correcter(spectrum)

def normalization(spectrum):
    normalizer = Normalizer()
    return normalizer(spectrum)

def trimming(spectrum, min_mz=2000, max_mz=20000):
    trimmer = Trimmer(min=min_mz, max=max_mz)
    return trimmer(spectrum)

def binning(spectrum):
    binner = Binner()
    return binner(spectrum)

# Function to interpolate spectrum to the target length
def interpolate_spectrum(spectrum, target_length=1000):
    """Interpolate spectrum to ensure the same length."""
    f = interpolate.interp1d(spectrum.mz, spectrum.intensity, kind='linear', fill_value="extrapolate")
    mz_new = np.linspace(spectrum.mz.min(), spectrum.mz.max(), target_length)
    intensity_new = f(mz_new)
    return mz_new, intensity_new

# Function to perform t-SNE visualization
def tsne_visualization(all_spectra, labels, species_names, output_filename, colors):
    # Dynamically adjust perplexity based on the number of samples
    n_samples = len(all_spectra)
    perplexity_value = min(n_samples - 1, 30)  # Use 30 if there are more than 30 samples, otherwise use n_samples - 1

    tsne = TSNE(n_components=2, perplexity=perplexity_value)
    tsne_results = tsne.fit_transform(np.array(all_spectra))
    
    # Map species names to colors
    species_labels = {species: idx for idx, species in enumerate(species_names)}
    numeric_labels = [species_labels[label] for label in labels]

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    for idx, species_name in enumerate(species_names):
        species_indices = [i for i, label in enumerate(labels) if label == species_name]
        species_spectra = [tsne_results[i] for i in species_indices]
        plt.scatter([spec[0] for spec in species_spectra], [spec[1] for spec in species_spectra], 
                    label=species_name, color=colors[idx])

    # Add legend, title, and labels
    plt.legend(title="Species")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization of Spectra")
    plt.savefig(output_filename)
    plt.close()
    print(f"t-SNE plot saved: {output_filename}")

# Main workflow for t-SNE before and after preprocessing
def tsne_before_and_after(species_names, base_dir_template, years, output_base_dir, target_length=1000):
    all_spectra_original = []
    all_spectra_preprocessed = []
    labels = []
    colors = ['#F0C501', '#E0389C', '#0AB086', '#C6E041', '#A559E5', '#6199E6', '#D68732', '#DB0059', '#9827E3', '#97D8E9']  # New colors

    for idx, species_name in enumerate(species_names):
        # Load spectra for the species from multiple years
        acqu_files, fid_files = load_spectra_multiple_years(species_name, base_dir_template, years)
        
        print(f"Loading spectra for {species_name}: {len(acqu_files)} files loaded.")
        
        # Process each spectrum for the species
        for acqu_file, fid_file in zip(acqu_files, fid_files):
            # Original spectrum
            spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)
            mz_original, intensity_original = interpolate_spectrum(spectrum, target_length)
            all_spectra_original.append(intensity_original)
            
            # Preprocessed spectrum
            intensity_preprocessed = preprocess_spectrum(acqu_file, fid_file, target_length)
            all_spectra_preprocessed.append(intensity_preprocessed)
            
            labels.append(species_name)

    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # t-SNE before preprocessing
    tsne_visualization(all_spectra_original, labels, species_names, os.path.join(output_base_dir, 'tsne_before_preprocessing.png'), colors)
    
    # t-SNE after preprocessing
    tsne_visualization(all_spectra_preprocessed, labels, species_names, os.path.join(output_base_dir, 'tsne_after_preprocessing.png'), colors)

# Running the workflow with multiple years
output_base_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/4_feature_extraction/scripts/full_ddbb/figures'  # You can specify your output directory
tsne_before_and_after(species_names, base_dir_template, years, output_base_dir)

print(f"t-SNE visualizations saved in {output_base_dir}")
