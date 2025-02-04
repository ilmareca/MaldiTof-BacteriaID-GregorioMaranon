import numpy as np
import matplotlib.pyplot as plt
from spectrum import SpectrumObject, Normalizer, VarStabilizer, Smoother, BaselineCorrecter, Trimmer
from sklearn.manifold import TSNE
from scipy import interpolate
import os

# Define paths and species data (you should update these paths with your actual file paths)
species_data = {
    'Escherichia_Coli': {
        'acqu_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18252920/0_D5/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18115713/0_E12/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18129745/0_A12/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18145755/0_B11/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18148292/0_F5/1/1SLin/acqu'
        ],
        'fid_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18252920/0_D5/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18115713/0_E12/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18129745/0_A12/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18145755/0_B11/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18148292/0_F5/1/1SLin/fid'
        ]
    },
    # Add more species here
    'Enterococcus_Faecalis': {
        'acqu_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18128679/0_D2/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18128154/0_B7/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18141809/0_B3/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18153673/0_C5/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18162817/0_E8/1/1SLin/acqu'
        ],
        'fid_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18128679/0_D2/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18128154/0_B7/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18141809/0_B3/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18153673/0_C5/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18162817/0_E8/1/1SLin/fid'
        ]
    },
    # Add more species here
    'Staphylococcus_Aureus': {
        'acqu_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18215596/0_B5/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18115713/0_F1/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18120280/0_A6/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18125381/0_D11/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18138077/0_A4/1/1SLin/acqu'
        ],
        'fid_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18215596/0_B5/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18115713/0_F1/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18120280/0_A6/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18125381/0_D11/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18138077/0_A4/1/1SLin/fid'
        ]
    },
    # Add more species here
    'Klebsiella_Pneumoniaee': {
        'acqu_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/16061980/0_E5/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/18117816/0_D9/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/18137525/0_A12/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/18146335/0_F1/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/18157589/0_C3/1/1SLin/acqu'
        ],
        'fid_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/16061980/0_E5/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/18117816/0_D9/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/18137525/0_A12/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/18146335/0_F1/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/18157589/0_C3/1/1SLin/fid'
        ]
    },
    # Add more species here
    'Staphylococcus_Epidermidis': {
        'acqu_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18115317/0_D4/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18115836/0_A12/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18116724/0_A7/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18128578/0_B5/1/1SLin/acqu',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18151897/0_G9/1/1SLin/acqu'
        ],
        'fid_files': [
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18115317/0_D4/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18115836/0_A12/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18116724/0_A7/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18128578/0_B5/1/1SLin/fid',
            '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18151897/0_G9/1/1SLin/fid'
        ]
    }
}

# Function to save spectrum image after each preprocessing step
def save_spectrum_image(spectrum, output_dir, species_name, filename, title):
    """Save the spectrum image after each preprocessing step."""
    plt.figure()
    plt.plot(spectrum.mz, spectrum.intensity)
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(f'{title} - {species_name}')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Step 1: Preprocessing (Variance stabilization, Smoothing, Baseline correction, Normalization, Trimming)
def preprocess_spectrum(spectrum, target_length=1000):
    # Step 1: Variance stabilization
    stabilizer = VarStabilizer(method="sqrt")
    spectrum = stabilizer(spectrum)

    # Interpolate spectrum to fixed length
    mz_new, intensity_new = interpolate_spectrum(spectrum, target_length)
    spectrum.mz = mz_new
    spectrum.intensity = intensity_new

    # Step 2: Smoothing
    smoother = Smoother(halfwindow=10, polyorder=3)
    spectrum = smoother(spectrum)
    mz_new, intensity_new = interpolate_spectrum(spectrum, target_length)
    spectrum.mz = mz_new
    spectrum.intensity = intensity_new

    # Step 3: Baseline correction
    baseline_correcter = BaselineCorrecter(method="SNIP", snip_n_iter=10)
    spectrum = baseline_correcter(spectrum)
    mz_new, intensity_new = interpolate_spectrum(spectrum, target_length)
    spectrum.mz = mz_new
    spectrum.intensity = intensity_new

    # Step 4: Normalization
    normalizer = Normalizer()
    spectrum = normalizer(spectrum)
    mz_new, intensity_new = interpolate_spectrum(spectrum, target_length)
    spectrum.mz = mz_new
    spectrum.intensity = intensity_new

    # Step 5: Trimming
    trimmer = Trimmer(min=2000, max=20000)
    spectrum = trimmer(spectrum)
    mz_new, intensity_new = interpolate_spectrum(spectrum, target_length)
    spectrum.mz = mz_new
    spectrum.intensity = intensity_new

    return spectrum

def interpolate_spectrum(spectrum, target_length=1000):
    """Interpolate spectrum to ensure the same length."""
    f = interpolate.interp1d(spectrum.mz, spectrum.intensity, kind='linear', fill_value="extrapolate")
    mz_new = np.linspace(spectrum.mz.min(), spectrum.mz.max(), target_length)
    intensity_new = f(mz_new)
    return mz_new, intensity_new

# Generate stacked plot of 5 preprocessed spectra for each species
def compare_spectra(species_data, output_base_dir):
    """
    Compare the 5 preprocessed spectra of each species in stacked format.
    """
    # Prepare figure for the comparison plot
    plt.figure(figsize=(10, 20))  # Adjust figure size to accommodate stacked spectra for each species
    colors = ['b', 'g', 'r', 'c', 'm']  # Colors for different species

    # Loop through each species and plot the 5 preprocessed spectra
    for idx, (species_name, paths) in enumerate(species_data.items()):
        acqu_files = paths['acqu_files']
        fid_files = paths['fid_files']

        # Create output directory for the species
        output_dir = os.path.join(output_base_dir, species_name)
        
        # Create subplot for the species
        plt.subplot(len(species_data), 1, idx + 1)  # Stack vertically for each species
        
        # Process the first 5 spectra for each species and plot them
        for i in range(5):
            acqu_file = acqu_files[i]
            fid_file = fid_files[i]

            # Load the spectrum
            spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)

            # Preprocess the spectrum (all 5 steps)
            processed_spectrum = preprocess_spectrum(spectrum)

            # Plot the preprocessed spectrum
            plt.plot(processed_spectrum.mz, processed_spectrum.intensity, label=f'Spectrum {i+1}', color=colors[i % len(colors)])
        
        # Label the plot
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.title(f'{species_name} - Stacked Spectra')
        plt.legend()

    # Save the final comparison plot
    os.makedirs(output_base_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, 'comparison_of_stacked_spectra.png'))
    plt.close()

# Running the full workflow
output_base_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/figures'

# Step 1: Compare spectra between species and generate comparison image
compare_spectra(species_data, output_base_dir)

print(f"All images saved in: {output_base_dir}")
# Generate overlaid plot of the first preprocessed spectrum for each species
def compare_first_spectra(species_data, output_base_dir):
    """
    Compare the first preprocessed spectrum of each species in an overlaid format.
    """
    # Prepare figure for the comparison plot
    plt.figure(figsize=(10, 6))  # Adjust figure size to accommodate overlaid spectra
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Colors for different species

    # Loop through each species and plot the first preprocessed spectrum
    for idx, (species_name, paths) in enumerate(species_data.items()):
        acqu_file = paths['acqu_files'][0]
        fid_file = paths['fid_files'][0]

        # Load the spectrum
        spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)

        # Preprocess the spectrum (all 5 steps)
        processed_spectrum = preprocess_spectrum(spectrum)

        # Plot the preprocessed spectrum
        plt.plot(processed_spectrum.mz, processed_spectrum.intensity, label=species_name, color=colors[idx % len(colors)])
    
    # Label the plot
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title('Comparison of First Preprocessed Spectrum of Each Species')
    plt.legend()

    # Save the final comparison plot
    os.makedirs(output_base_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, 'comparison_of_first_spectra.png'))
    plt.close()

# Step 2: Compare the first spectrum of each species and generate comparison image
compare_first_spectra(species_data, output_base_dir)