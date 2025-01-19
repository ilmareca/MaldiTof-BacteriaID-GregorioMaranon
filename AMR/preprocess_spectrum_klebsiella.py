import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import zipfile
from spectrum import SpectrumObject, VarStabilizer, Smoother, BaselineCorrecter, Normalizer, Trimmer
from scipy import interpolate
import joblib
import tempfile

# Define the base directory
base_dir = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/klebsiellaPneumoniae'
preprocessed_dir = './amr'

# Create logs directory if it doesn't exist
os.makedirs('./logs/amr', exist_ok=True)

# Configure loggers
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(f'./logs/amr/error_log_{timestamp}.log')
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)

fid_acqu_logger = logging.getLogger('fid_acqu_logger')
fid_acqu_handler = logging.FileHandler(f'./logs/amr/fid_acqu_log_{timestamp}.log')
fid_acqu_logger.addHandler(fid_acqu_handler)
fid_acqu_logger.setLevel(logging.INFO)

def load_spectra_from_dir(base_dir):
    acqu_files = []
    fid_files = []
    labels = []
    
    for genus in os.listdir(base_dir):
        genus_dir = os.path.join(base_dir, genus)
        if os.path.isdir(genus_dir):
            for species in os.listdir(genus_dir):
                species_dir = os.path.join(genus_dir, species)
                if os.path.isdir(species_dir):
                    for extern_id in os.listdir(species_dir):
                        extern_id_dir = os.path.join(species_dir, extern_id)
                        if os.path.isdir(extern_id_dir):
                            for target_position in os.listdir(extern_id_dir):
                                target_position_dir = os.path.join(extern_id_dir, target_position)
                                if os.path.isdir(target_position_dir):
                                    for measure in os.listdir(target_position_dir):
                                        measure_dir = os.path.join(target_position_dir, measure)
                                        if os.path.isdir(measure_dir):
                                            slin_dir = os.path.join(measure_dir, '1SLin')
                                            if os.path.isdir(slin_dir):
                                                acqu_path = os.path.join(slin_dir, 'acqu')
                                                fid_path = os.path.join(slin_dir, 'fid')
                                                if os.path.exists(acqu_path) and os.path.exists(fid_path):
                                                    fid_acqu_logger.info(f"Found acqu file: {acqu_path}")
                                                    fid_acqu_logger.info(f"Found fid file: {fid_path}")
                                                    acqu_files.append(acqu_path)
                                                    fid_files.append(fid_path)
                                                    labels.append(f"{genus}_{species}_{extern_id}")

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
def preprocess_and_save_spectra(base_dir, target_length=1000, preprocessed_dir=None):
    acqu_files, fid_files, labels = load_spectra_from_dir(base_dir)
    all_spectra = []

    for acqu_file, fid_file, label in zip(acqu_files, fid_files, labels):
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
    preprocess_and_save_spectra(base_dir, target_length, preprocessed_dir)

if __name__ == "__main__":
    main()