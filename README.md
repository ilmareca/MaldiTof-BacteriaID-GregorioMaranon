# Project: MALDI-TOF Spectra Analysis for Bacterial Identification and Antibiotic Resistance Prediction

This project focuses on the analysis of bacterial spectra obtained using MALDI-TOF mass spectrometry. The goal is to predict both the bacterial species and genus, with a specific focus on predicting antibiotic resistance for the **Klebsiella pneumoniae** bacterium. The pipeline applies to multiple datasets, including those from the **Hospital General Universitario Gregorio Marañón (HGUGM)** and publicly available databases like **DRIAMS** and **RKI**. These external datasets are used to validate and improve prediction accuracy across various bacterial strains.

---

## 📁 Project Structure

1. **1_raw_data/**: Contains the original raw data from different sources (HGUGM, DRIAMS, RKI).
2. **2_data_cleaning/**: Contains scripts and outputs for cleaning the raw spectra and preparing them for analysis.
3. **3_data_statistics/**: Contains exploratory data analysis (EDA), statistics, and visual summaries of the bacterial spectra.
4. **4_data_preprocessing/**: Contains scripts for feature extraction, normalization, and preparation for model training.
5. **5_modeling/**: Contains scripts for model training and evaluation, including bacterial species prediction and resistance classification.
6. **tests/**: Contains unit tests to ensure the reliability of the entire pipeline.

---

## 🛠️ Pipeline Steps

1. **Raw Data Loading**: The raw spectral data is collected from **HGUGM** and public datasets (**DRIAMS** and **RKI**), located in `1_raw_data/raw_files`.
2. **Data Cleaning**: Cleaning and noise removal are applied to the spectra using scripts in `2_data_cleaning/scripts`.
3. **Exploratory Data Analysis (EDA)**: Statistical summaries and visualizations are generated in `3_data_statistics/scripts` to understand the data distribution and patterns.
4. **Preprocessing and Feature Engineering**: Spectral preprocessing, feature extraction, and transformations are performed in `4_data_preprocessing/scripts`.
5. **Model Training and Evaluation**: Machine learning models are trained in `5_modeling/scripts` to:
    - Predict the bacterial **genus** and **species**.
    - Predict the **antibiotic resistance status** for **Klebsiella pneumoniae** strains.

---

## 🧪 **Datasets**

- **Hospital General Universitario Gregorio Marañón (HGUGM)**: Clinical bacterial spectra obtained from MALDI-TOF mass spectrometry, specifically focusing on **Klebsiella pneumoniae** samples.
- **DRIAMS Database**: Publicly available mass spectrometry data from various bacterial strains, used for external validation.
- **RKI Database**: Spectra and metadata provided by the Robert Koch Institute for comparative analysis and prediction refinement.

---

## 📜 Environment Requirements

- Python [version]
- Required libraries: 
  ```bash
  pip install -r requirements.txt
