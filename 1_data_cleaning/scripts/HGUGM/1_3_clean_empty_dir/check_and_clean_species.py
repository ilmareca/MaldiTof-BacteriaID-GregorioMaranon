import os
import pandas as pd
import logging

# Configurar el registro
log_dir = f'/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/logs/HGUGM/check-and-clean-species'

os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'zero_count_paths.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

def delete_empty_dirs(path):
    if not os.listdir(path):
        os.rmdir(path)
        logging.info(f"Deleted empty directory: {path}")
        print(f"Deleted empty directory: {path}")

def count_spectra(base_dir, years):
    data = {}
    try:
        for year in years:
            year_dir = os.path.join(base_dir, str(year))
            if os.path.isdir(year_dir):
                matched_bacteria_dir = os.path.join(year_dir, 'matched_bacteria')
                for genus in os.listdir(matched_bacteria_dir):
                    genus_dir = os.path.join(matched_bacteria_dir, genus)
                    if os.path.isdir(genus_dir):
                        for species in os.listdir(genus_dir):
                            species_dir = os.path.join(genus_dir, species)
                            if os.path.isdir(species_dir):
                                count = 0
                                for extern_id in os.listdir(species_dir):
                                    extern_id_dir = os.path.join(species_dir, extern_id)
                                    if os.path.isdir(extern_id_dir):
                                        for target_position in os.listdir(extern_id_dir):
                                            target_position_dir = os.path.join(extern_id_dir, target_position)
                                            if os.path.isdir(target_position_dir):
                                                maldis = os.listdir(target_position_dir)
                                                count += len(maldis)
                                                delete_empty_dirs(target_position_dir)
                                        delete_empty_dirs(extern_id_dir)
                                species_name = f"{genus} {species}"
                                if species_name in data:
                                    data[species_name] += count
                                else:
                                    data[species_name] = count
                                if count == 0:
                                    logging.info(f"Zero count for path: {species_dir}")
                                print(f"Added/Updated species: {species_name} with count {count}")
                                delete_empty_dirs(species_dir)
                        delete_empty_dirs(genus_dir)
                delete_empty_dirs(matched_bacteria_dir)
    except Exception as e:
        logging.error(f"Error in count_spectra: {e}")
    return data

def main():
    base_dir = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2'
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    output_csv = 'species_spectra_count.csv'
    
    # Contar los espectros de cada especie
    data = count_spectra(base_dir, years)
    
    # Crear un DataFrame
    df = pd.DataFrame(list(data.items()), columns=['genus+species', 'count'])
    
    # Ordenar el DataFrame de menor a mayor por la columna 'count'
    df = df.sort_values(by='count')
    
    # Guardar el DataFrame en un archivo CSV
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    main()