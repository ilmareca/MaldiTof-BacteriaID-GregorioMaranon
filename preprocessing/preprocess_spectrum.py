import os
import numpy as np
import pandas as pd
import zipfile
from spectrum import SpectrumObject, VarStabilizer, Smoother, BaselineCorrecter, Normalizer, Trimmer
from scipy import interpolate
import joblib

# Define the base directory and years
base_dir_template = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2'
preprocessed_dir = './preprocessed_spectra'
years = [2018, 2019, 2020, 2021, 2022, 2023]

def load_spectra_from_zip(base_dir_template, years):
    acqu_files = []
    fid_files = []
    labels = []
    
    for year in years:
        base_dir = base_dir_template.format(year=year)
        for file in os.listdir(base_dir):
            if file.endswith(".zip"):
                print(f"Processing {file}...")
                with zipfile.ZipFile(os.path.join(base_dir, file), 'r') as zip_ref:
                    for genus in zip_ref.namelist():
                        if genus.endswith('/'):
                            print(f"Processing {genus}...")
                            for species in zip_ref.namelist():
                                if species.startswith(genus) and species.endswith('/'):
                                    print(f"Processing {species}...")
                                    for extern_id in zip_ref.namelist():
                                        if extern_id.startswith(species) and extern_id.endswith('/'):
                                            print(f"Processing {extern_id}...")
                                            for target_position in zip_ref.namelist():
                                                if target_position.startswith(extern_id) and target_position.endswith('/'):
                                                    print(f"Processing {target_position}...")
                                                    for measure in zip_ref.namelist():
                                                        if measure.startswith(target_position) and measure.endswith('/'):
                                                            print(f"Processing {measure}...")
                                                            slin_dir = measure + '1SLin/'
                                                            for file_info in zip_ref.infolist():
                                                                if file_info.filename.startswith(slin_dir):
                                                                    if file_info.filename.endswith("acqu"):
                                                                        print(f"Found acqu file: {file_info.filename}")
                                                                        acqu_files.append(zip_ref.extract(file_info, path=base_dir))
                                                                        labels.append(f"{genus}_{species}")
                                                                    elif file_info.filename.endswith("fid"):
                                                                        fid_files.append(zip_ref.extract(file_info, path=base_dir))

    return acqu_files, fid_files, labels

# Function to preprocess spectra with variance stabilization, smoothing, baseline correction, normalization, and trimming
def preprocess_spectrum(acqu_file, fid_file, target_length=1000):
    spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)
    
    # Apply each preprocessing step
    spectrum = variance_stabilization(spectrum)
    spectrum = smoothing(spectrum)
    spectrum = baseline_correction(spectrum)
    spectrum = normalization(spectrum)
    spectrum = trimming(spectrum)
    
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

# Function to interpolate spectrum to the target length
def interpolate_spectrum(spectrum, target_length=1000):
    """Interpolate spectrum to ensure the same length."""
    f = interpolate.interp1d(spectrum.mz, spectrum.intensity, kind='linear', fill_value="extrapolate")
    mz_new = np.linspace(spectrum.mz.min(), spectrum.mz.max(), target_length)
    intensity_new = f(mz_new)
    return mz_new, intensity_new

# Function to preprocess and save all spectra
def preprocess_and_save_spectra(base_dir_template, years, target_length=1000, preprocessed_dir=None):
    acqu_files, fid_files, labels = load_spectra_from_zip(base_dir_template, years)
    all_spectra = []

    for acqu_file, fid_file, label in zip(acqu_files, fid_files, labels):
        print(f"Preprocessing spectrum for {label}...")
        intensity_new = preprocess_spectrum(acqu_file, fid_file, target_length)
        all_spectra.append(intensity_new)

    X = np.array(all_spectra)
    y = np.array(labels)

    if preprocessed_dir:
        os.makedirs(preprocessed_dir, exist_ok=True)
        joblib.dump(X, os.path.join(preprocessed_dir, 'X.pkl'))
        joblib.dump(y, os.path.join(preprocessed_dir, 'y.pkl'))

    return X, y

def main():
    target_length = 1000  # Length to which spectra will be interpolated
    
    # Preprocess and save spectra
    preprocess_and_save_spectra(base_dir_template, years, target_length, preprocessed_dir)

if __name__ == "__main__":
    main()