import os
import csv
import xml.etree.ElementTree as ET
import logging

# Configurar el logger
logging.basicConfig(filename='/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/2_data_statistics/logs/HGUGM/error_generate_bacteria_statistics.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def count_extern_ids_and_maldis_hgugm(base_dir, extern_id_log_file, maldis_log_file):
    extern_id_count = 0
    maldis_count = 0
    try:
        with open(extern_id_log_file, 'w') as extern_id_log, open(maldis_log_file, 'w') as maldis_log:
            for year in os.listdir(base_dir):
                if len(year) == 4 and year.isdigit() and os.path.isdir(os.path.join(base_dir, year)):
                    year_dir = os.path.join(base_dir, year)
                    if os.path.isdir(year_dir):
                        matched_bacteria_dir = os.path.join(year_dir, 'matched_bacteria')
                        for genus in os.listdir(matched_bacteria_dir):
                            genus_dir = os.path.join(matched_bacteria_dir, genus)
                            if os.path.isdir(genus_dir):
                                for species in os.listdir(genus_dir):
                                    species_dir = os.path.join(genus_dir, species)
                                    if os.path.isdir(species_dir):
                                        extern_ids = os.listdir(species_dir)
                                        for extern_id in extern_ids:
                                            extern_id_count += 1
                                            extern_id_dir = os.path.join(species_dir, extern_id)
                                            extern_id_log.write(f"Found file {extern_id_dir}, current count: {extern_id_count}\n")
                                            if os.path.isdir(extern_id_dir):
                                                for target_position in os.listdir(extern_id_dir):
                                                    target_position_dir = os.path.join(extern_id_dir, target_position)
                                                    if os.path.isdir(target_position_dir):
                                                        maldis = os.listdir(target_position_dir)
                                                        for maldi in maldis:
                                                            maldis_count += 1
                                                            maldi_dir = os.path.join(target_position_dir, maldi)
                                                            maldis_log.write(f"Found maldi in {maldi_dir}, current count: {maldis_count}\n")
    except Exception as e:
        logging.error(f"Error in count_extern_ids_and_maldis_hgugm: {e}")
    return extern_id_count, maldis_count

def count_extern_ids_and_maldis_rki(base_dir, extern_id_log_file, maldis_log_file):
    extern_id_count = 0
    maldis_count = 0
    try:
        with open(extern_id_log_file, 'w') as extern_id_log, open(maldis_log_file, 'w') as maldis_log:
            for genus in os.listdir(base_dir):
                genus_dir = os.path.join(base_dir, genus)
                if os.path.isdir(genus_dir):
                    for species in os.listdir(genus_dir):
                        species_dir = os.path.join(genus_dir, species)
                        if os.path.isdir(species_dir):
                            extern_ids = os.listdir(species_dir)
                            for extern_id in extern_ids:
                                extern_id_count += 1
                                extern_id_dir = os.path.join(species_dir, extern_id)
                                extern_id_log.write(f"Found file {extern_id_dir}, current count: {extern_id_count}\n")
                                if os.path.isdir(extern_id_dir):
                                    for messung in os.listdir(extern_id_dir):
                                        messung_dir = os.path.join(extern_id_dir, messung)
                                        if os.path.isdir(messung_dir):
                                            for target_position in os.listdir(messung_dir):
                                                target_position_dir = os.path.join(messung_dir, target_position)
                                                if os.path.isdir(target_position_dir):
                                                    maldis = os.listdir(target_position_dir)
                                                    for maldi in maldis:
                                                        maldis_count += 1
                                                        maldi_dir = os.path.join(target_position_dir, maldi)
                                                        maldis_log.write(f"Found maldi in {maldi_dir}, current count: {maldis_count}\n")

    except Exception as e:
        logging.error(f"Error in count_extern_ids_and_maldis_rki: {e}")
    return extern_id_count, maldis_count

def count_maldis_driams(base_dir, log_file):
    count = 0
    try:
        with open(log_file, 'w') as log:
            for driams in ['DRIAMS_A', 'DRIAMS_B', 'DRIAMS_C', 'DRIAMS_D']:
                driams_dir = os.path.join(base_dir, driams, 'TOTAL')
                for genus in os.listdir(driams_dir):
                    genus_dir = os.path.join(driams_dir, genus)
                    if os.path.isdir(genus_dir):
                        for species in os.listdir(genus_dir):
                            species_dir = os.path.join(genus_dir, species)
                            if os.path.isdir(species_dir):
                                files = os.listdir(species_dir)
                                for fid in files:
                                    fid_dir = os.path.join(species_dir, fid)
                                    count += 1
                                    log.write(f"Found maldi in {fid_dir}, current count: {count}\n")
    except Exception as e:
        logging.error(f"Error in count_maldis_driams: {e}")
    return count

def count_genus_species(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        genus_count = len(root.findall(".//Genus"))
        species_count = len(root.findall(".//Species"))
        return genus_count, species_count
    except Exception as e:
        logging.error(f"Error in count_genus_species: {e}")
        return 0, 0

def get_top_species_hgugm(xml_file, top_n=5):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        species_list = []
        for genus in root.findall(".//Genus"):
            genus_name = genus.attrib['name']
            for species in genus.findall(".//Species"):
                species_name = species.attrib['name']
                total = int(species.find('Total').text)
                species_list.append((genus_name, species_name, total))
        species_list.sort(key=lambda x: x[2], reverse=True)
        return species_list[:top_n]
    except Exception as e:
        logging.error(f"Error in get_top_species_hgugm: {e}")
        return []

def get_top_species_driams_rki(xml_file, top_n=5):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        species_list = []
        for genus in root.findall(".//Genus"):
            genus_name = genus.attrib['name']
            for species in genus.findall(".//Species"):
                species_name = species.attrib['name']
                total = int(species.text)
                species_list.append((genus_name, species_name, total))
        species_list.sort(key=lambda x: x[2], reverse=True)
        return species_list[:top_n]
    except Exception as e:
        logging.error(f"Error in get_top_species_driams_rki: {e}")
        return []

def process_hgugm(base_dir, xml_file, csv_writer):
    try:
        dataset = "HGUGM"
        muestras, maldis = count_extern_ids_and_maldis_hgugm(base_dir, "/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/2_data_statistics/logs/HGUGM/count_of_externID_HGUGM.log", "/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/2_data_statistics/logs/HGUGM/count_of_Maldis_HGUGM.log")
        genus, species = count_genus_species(xml_file)
        top_species = get_top_species_hgugm(xml_file)

        csv_writer.writerow([dataset, muestras, maldis, genus, species])
        for genus_name, species_name, total in top_species:
            csv_writer.writerow([dataset, "", "", "", "", f"{genus_name} {species_name}", total, 0])
    except Exception as e:
        logging.error(f"Error in process_hgugm: {e}")

def process_driams(base_dir, csv_writer):
    try:
        dataset = "DRIAMS"
        muestras = "unknown"
        maldis = count_maldis_driams(base_dir, "/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/2_data_statistics/logs/DRIAMS/count_of_Maldis_DRIAMS.log")
        combined_xml = os.path.join(base_dir, "combined_driams.xml")
        genus, species = count_genus_species(combined_xml)
        top_species = get_top_species_driams_rki(combined_xml)

        csv_writer.writerow([dataset, muestras, maldis, genus, species])
        for genus_name, species_name, total in top_species:
            csv_writer.writerow([dataset, "", "", "", "", f"{genus_name} {species_name}", total, 0])
    except Exception as e:
        logging.error(f"Error in process_driams: {e}")

def process_rki(base_dir, xml_file, csv_writer):
    try:
        dataset = "RKI"
        muestras, maldis = count_extern_ids_and_maldis_rki(base_dir, "/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/2_data_statistics/logs/RKI/count_of_externID_RKI.log", "/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/2_data_statistics/logs/RKI/count_of_Maldis_RKI.log")
        genus, species = count_genus_species(xml_file)
        top_species = get_top_species_driams_rki(xml_file)

        csv_writer.writerow([dataset, muestras, maldis, genus, species])
        for genus_name, species_name, total in top_species:
            csv_writer.writerow([dataset, "", "", "", "", f"{genus_name} {species_name}", total, 0])
    except Exception as e:
        logging.error(f"Error in process_rki: {e}")

def main():
    try:
        with open('/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/2_data_statistics/outputs/global_stat.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["dataset", "#samples", "#maldis", "#genus", "#species", "genus+species", "#fids", "AMR"])

            # Process HGUGM
            process_hgugm('/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2', '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/aggregated_report.xml', csv_writer)

            # Process DRIAMS
            process_driams('/export/data_ml4ds/bacteria_id/relevant_datasets/DRIAMS_PROCESSED_DATABASE', csv_writer)

            # Process RKI
            process_rki('/export/data_ml4ds/bacteria_id/relevant_datasets/10.5281/RKI_ROOT', '/export/data_ml4ds/bacteria_id/relevant_datasets/RKI_STATS/report.xml', csv_writer)
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()