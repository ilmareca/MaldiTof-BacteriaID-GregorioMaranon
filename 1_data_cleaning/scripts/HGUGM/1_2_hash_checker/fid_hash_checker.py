import os
import hashlib
import pandas as pd
import shutil

def process_fid(fid_path, empty_log, hash_log, error_log, df):
    try:
        print(f"Processing file: {fid_path}")
        # Load the FID file in binary mode
        with open(fid_path, 'rb') as f:
            content = f.read()

        # Convert the content to hexadecimal
        hex_content = content.hex()

        # Check if the content is all zeros
        if set(hex_content.strip()) == {'0'}:
            with open(empty_log, 'a') as log:
                log.write(f"{fid_path} está vacío.\n")
            # Eliminar el archivo y las carpetas correspondientes
            target_position_dir = os.path.dirname(fid_path)
            shutil.rmtree(target_position_dir)
            extern_id_dir = os.path.dirname(target_position_dir)
            if not os.listdir(extern_id_dir):
                shutil.rmtree(extern_id_dir)
                species_dir = os.path.dirname(extern_id_dir)
                if not os.listdir(species_dir):
                    shutil.rmtree(species_dir)
                    genus_dir = os.path.dirname(species_dir)
                    if not os.listdir(genus_dir):
                        shutil.rmtree(genus_dir)
        else:
            # Generate a 64-character hash
            hash_value = hashlib.sha256(content).hexdigest()
            # Check if the hash already exists in the DataFrame
            if hash_value in df['hash'].values:
                existing_path = df[df['hash'] == hash_value]['path'].values[0]
                with open(hash_log, 'a') as log:
                    log.write(f"{fid_path} and {existing_path} have the same hash.\n")
                print(f"{fid_path} and {existing_path} have the same hash.")
            else:
                # Agregar el hash y la ubicación al DataFrame
                new_row = pd.DataFrame({'hash': [hash_value], 'path': [fid_path]})
                df = pd.concat([df, new_row], ignore_index=True)
    except Exception as e:
        with open(error_log, 'a') as log:
            log.write(f"Error processing {fid_path}: {str(e)}\n")
        print(f"Error processing {fid_path}: {str(e)}")
    return df

def main(year):
    base_dir = f'/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/{year}/matched_bacteria'
    log_dir = f'/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/logs/HGUGM/fid-hash-checker'
    hashes_dir = f'/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/{year}'
    os.makedirs(log_dir, exist_ok=True)
    empty_log = os.path.join(log_dir, 'empty_fid.log')
    hash_log = os.path.join(log_dir, 'fid_with_same_hash.log')
    error_log = os.path.join(log_dir, 'error.log')
    df = pd.DataFrame(columns=['hash', 'path'])

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
                                print(f"Processing target_position: {target_position}")
                                if os.path.isdir(target_position_dir):
                                    for measure in os.listdir(target_position_dir):
                                        measure_dir = os.path.join(target_position_dir, measure)
                                        if os.path.isdir(measure_dir):
                                            for sample in os.listdir(measure_dir):
                                                sample_dir = os.path.join(measure_dir, sample)
                                                if os.path.isdir(sample_dir):
                                                    for fid in os.listdir(sample_dir):
                                                        fid_path = os.path.join(sample_dir, fid)
                                                        if os.path.isfile(fid_path) and fid == 'fid':
                                                            print(f"Processing FID: {fid_path}")
                                                            df = process_fid(fid_path, empty_log, hash_log, error_log, df)
                                                        
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(hashes_dir, 'fid_hashes.csv'), index=False)
    print("Process completed. The results have been saved in 'fid_hashes.csv'.")

if __name__ == "__main__":
    year = input("Please enter the year you want to clean (e.g., 2020): ").strip()

    main(year)
