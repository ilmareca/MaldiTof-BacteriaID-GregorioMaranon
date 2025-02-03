import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import zipfile
from spectrum import SpectrumObject, VarStabilizer, Smoother, BaselineCorrecter, Normalizer, Trimmer, Binner
from scipy import interpolate
import joblib
import tempfile

# Define the base directory and years
base_dir_template = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/MaldiMaranonDB'
preprocessed_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/scripts/outputs'
years = [2018, 2019, 2020, 2021, 2022, 2023]

# Configure loggers
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(f'/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/logs/full_ddbb/error_log_{timestamp}.log')
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)

zip_logger = logging.getLogger('zip_logger')
zip_handler = logging.FileHandler(f'/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/logs/full_ddbb/zip_log_{timestamp}.log')
zip_logger.addHandler(zip_handler)
zip_logger.setLevel(logging.INFO)

fid_acqu_logger = logging.getLogger('fid_acqu_logger')
fid_acqu_handler = logging.FileHandler(f'/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/logs/full_ddbb/fid_acqu_log_{timestamp}.log')
fid_acqu_logger.addHandler(fid_acqu_handler)
fid_acqu_logger.setLevel(logging.INFO)


def load_spectra_from_dir(base_dir_template, years):
    acqu_files = []
    fid_files = []
    labels = []
    processed_paths = set()
    spectra_count = 0
    
    for year in years:
        base_dir = os.path.join(base_dir_template, str(year))
        zip_logger.info(f"Processing {base_dir}...")
        for genus in os.listdir(base_dir):
            genus_dir = os.path.join(base_dir, genus)
            if os.path.isdir(genus_dir) and genus_dir not in processed_paths:
                processed_paths.add(genus_dir)
                for species in os.listdir(genus_dir):
                    species_dir = os.path.join(genus_dir, species)
                    if os.path.isdir(species_dir) and species_dir not in processed_paths:
                        processed_paths.add(species_dir)
                        for extern_id in os.listdir(species_dir):
                            extern_id_dir = os.path.join(species_dir, extern_id)
                            if os.path.isdir(extern_id_dir) and extern_id_dir not in processed_paths:
                                processed_paths.add(extern_id_dir)
                                for target_position in os.listdir(extern_id_dir):
                                    target_position_dir = os.path.join(extern_id_dir, target_position)
                                    if os.path.isdir(target_position_dir) and target_position_dir not in processed_paths:
                                        processed_paths.add(target_position_dir)
                                        for measure in os.listdir(target_position_dir):
                                            measure_dir = os.path.join(target_position_dir, measure)
                                            if os.path.isdir(measure_dir) and measure_dir not in processed_paths:
                                                processed_paths.add(measure_dir)
                                                slin_dir = os.path.join(measure_dir, '1SLin')
                                                if os.path.isdir(slin_dir) and slin_dir not in processed_paths:
                                                    processed_paths.add(slin_dir)
                                                    acqu_path = os.path.join(slin_dir, 'acqu')
                                                    fid_path = os.path.join(slin_dir, 'fid')
                                                    if os.path.exists(acqu_path) and os.path.exists(fid_path):
                                                        try:
                                                            fid_acqu_logger.info(f"Found acqu file: {acqu_path}")
                                                            fid_acqu_logger.info(f"Found fid file: {fid_path}")
                                                            acqu_files.append(acqu_path)
                                                            fid_files.append(fid_path)
                                                            labels.append(f"{genus}_{species}_{extern_id}")
                                                            spectra_count += 1
                                                            #if spectra_count >= max_spectra:
                                                            #   return acqu_files, fid_files, labels
                                                        except Exception as e:
                                                            error_logger.error(f"Error processing {acqu_path} and {fid_path}: {e}")
    return acqu_files, fid_files, labels

# Function to preprocess spectra with variance stabilization, smoothing, baseline correction, normalization, and trimming
def preprocess_spectrum(acqu_file, fid_file):
    spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)
    
    # Apply each preprocessing step
    spectrum = variance_stabilization(spectrum)
    spectrum = smoothing(spectrum)
    spectrum = baseline_correction(spectrum)
    spectrum = normalization(spectrum)
    spectrum = trimming(spectrum)
    spectrum = binning(spectrum)
    
    return spectrum.intensity


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


# Function to preprocess and save all spectra
def preprocess_and_save_spectra(base_dir_template, years, preprocessed_dir=None):
    acqu_files, fid_files, labels = load_spectra_from_dir(base_dir_template, years)
    all_spectra = []

    for acqu_file, fid_file, label in zip(acqu_files, fid_files, labels):
        intensity_new = preprocess_spectrum(acqu_file, fid_file)
        all_spectra.append(intensity_new)

    X = np.array(all_spectra)
    y = np.array(labels)

    if preprocessed_dir:
        os.makedirs(preprocessed_dir, exist_ok=True)
        joblib.dump(X, os.path.join(preprocessed_dir, 'X.pkl'))
        joblib.dump(y, os.path.join(preprocessed_dir, 'y.pkl'))

    return X, y

def main():
    
    # Preprocess and save spectra
    preprocess_and_save_spectra(base_dir_template, years, preprocessed_dir)

if __name__ == "__main__":
    main()