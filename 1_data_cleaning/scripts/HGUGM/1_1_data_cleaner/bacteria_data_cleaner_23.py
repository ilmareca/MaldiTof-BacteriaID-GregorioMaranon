import os
import zipfile
import xml.etree.ElementTree as ET
import logging
import pandas as pd
import re
import json
from datetime import datetime


THRESHOLD = 1.7

def setup_logging(year):
    """
    Sets up logging with file names including the specified year.

    :param year: The year to include in the log file names.
    """
    # Create the directory including the year if it doesn't exist
    log_directory = f'/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/1_data_cleaning/logs/HGUGM/2023'
    os.makedirs(log_directory, exist_ok=True)

    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Setup individual loggers for different logs
    xml_found_log = logging.getLogger('xml_found_log')
    xml_found_log_handler = logging.FileHandler(os.path.join(log_directory, f'xml_found_log_2023_{timestamp}.log'))
    xml_found_log.addHandler(xml_found_log_handler)

    analyte_validation_log = logging.getLogger('analyte_validation_log')
    analyte_validation_log_handler = logging.FileHandler(os.path.join(log_directory, f'analyte_validation_log_2023_{timestamp}.log'))
    analyte_validation_log.addHandler(analyte_validation_log_handler)
    
    analyte_zip_found_log = logging.getLogger('analyte_zip_found_log')
    analyte_zip_found_log_handler = logging.FileHandler(os.path.join(log_directory, f'analyte_zip_found_log_2023_{timestamp}.log'))
    analyte_zip_found_log.addHandler(analyte_zip_found_log_handler)
    
    directory_move_log = logging.getLogger('directory_move_log')
    directory_move_log_handler = logging.FileHandler(os.path.join(log_directory, f'directory_move_log_2023_{timestamp}.log'))
    directory_move_log.addHandler(directory_move_log_handler)

    # Logger for recording errors that occur during processing
    error_log = logging.getLogger('error_log')
    error_log_handler = logging.FileHandler(os.path.join(log_directory, f'error_log_2023_{timestamp}.log'))
    error_log.addHandler(error_log_handler)

    # Set log level and format
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Return the loggers for use in the main code
    return xml_found_log, analyte_validation_log, analyte_zip_found_log, directory_move_log, error_log

def process_analyte(analyte, analyte_validation_log):
    """
    Processes an analyte element to extract genus, species, extern_id, and target_position.

    :param analyte: The analyte XML element.
    :param analyte_validation_log: Logger for validation messages.
    :return: Tuple of (genus, species, extern_id, target_position) if valid, otherwise (None, None, None, None).
    """
    extern_id = analyte.attrib.get("externId")
    target_position = analyte.attrib.get("targetPosition")
    msp_matches = analyte.findall(".//MspMatch")[:3]

    # Validate required attributes
    if not extern_id or not target_position or not msp_matches:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Missing attributes")
        return None, None, None, None
    
    if len(extern_id) < 8 or not extern_id[:8].isdigit():
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Invalid extern_id")
        return None, None, None, None

    # Extract the referencePatternName and globalMatchValue from the first MspMatch
    ref_pattern_1 = msp_matches[0].attrib.get("referencePatternName")
    global_match_value_1 = float(msp_matches[0].attrib.get("globalMatchValue", "0"))

    if global_match_value_1 < THRESHOLD:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Low global_match_value ({global_match_value_1})")
        return None, None, None, None

    genus_1, species_1 = ref_pattern_1.split()[:2]               
    species_1 = re.sub('[^a-zA-Z0-9 \n\.]', ' ', species_1).split(" ")[0]
    if not re.match(r'^[a-zA-Z]+$', species_1):
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Invalid species name (1: {species_1})")
        return None, None, None, None
    
    # Check the second match
    ref_pattern_2 = msp_matches[1].attrib.get("referencePatternName")
    genus_2, species_2 = ref_pattern_2.split()[:2]
    species_2 =  re.sub('[^a-zA-Z0-9 \n\.]', ' ', species_2).split(" ")[0]
    if not re.match(r'^[a-zA-Z]+$', species_2):
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Invalid species name (2: {species_2})")
        return None, None, None, None
    global_match_value_2 = float(msp_matches[1].attrib.get("globalMatchValue", "0"))

    if genus_1 != genus_2 and global_match_value_2 >= THRESHOLD:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Genus mismatch (1: {genus_1}, 2: {genus_2}) with high global_match_value ({global_match_value_2})")
        return None, None, None, None
    
    if species_1 != species_2 and global_match_value_2 >= THRESHOLD:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Species mismatch at high global_match_value (1: {species_1}, 2: {species_2}): {global_match_value_2}")
        return None, None, None, None
    
    if species_1 != species_2 and global_match_value_2 < THRESHOLD:
        analyte_validation_log.info(f"⚠️ extern_id = {extern_id} at {target_position} accepted: Species mismatch at low global_match_value (1: {species_1}, 2: {species_2}): {global_match_value_2}")
    
    # Check the third match
    ref_pattern_3 = msp_matches[2].attrib.get("referencePatternName")
    genus_3, species_3 = ref_pattern_3.split()[:2]
    species_3 =  re.sub('[^a-zA-Z0-9 \n\.]', ' ', species_3).split(" ")[0]
    if not re.match(r'^[a-zA-Z]+$', species_3):
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Invalid species name (3: {species_3})")
        return None, None, None, None
    global_match_value_3 = float(msp_matches[2].attrib.get("globalMatchValue", "0"))

    if genus_1 != genus_3 and global_match_value_3 >= THRESHOLD:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Genus mismatch (1: {genus_1}, 3: {genus_3}) with high global_match_value ({global_match_value_2})")
        return None, None, None, None

    if species_1 != species_3 and global_match_value_3 >= THRESHOLD:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Species mismatch at high global_match_value (1: {species_1}, 3: {species_3}): {global_match_value_3}")
        return None, None, None, None

    if species_1 != species_3 and global_match_value_3 < THRESHOLD:
        analyte_validation_log.info(f"⚠️ extern_id = {extern_id} at {target_position} accepted: Species mismatch at low global_match_value (1: {species_1}, 3: {species_3}): {global_match_value_3}")

    # Return the genus and species from the first match, and the externId
    analyte_validation_log.info(f"✅ extern_id = {extern_id} at {target_position} accepted: {genus_1} {species_1} with {global_match_value_1}")
    return genus_1, species_1, extern_id, target_position

def zip_and_move_folder(zip_ref: zipfile.ZipFile, folder_name, new_dest_path, directory_move_log, error_log):
    """
    Extracts and moves a folder from a zip file to a new destination.

    :param zip_ref: Reference to the zip file.
    :param folder_name: Name of the folder to move.
    :param new_dest_path: Destination path.
    :param directory_move_log: Logger for directory move messages.
    :param error_log: Logger for error messages.
    """
    for src_file_path in zip_ref.namelist():
        if not src_file_path.startswith(folder_name):
            continue
        if src_file_path.endswith('/'):
            continue
        dest_file_path = os.path.join(new_dest_path, src_file_path[len(folder_name):])

        success = False
        for _ in range(3):
            try:
                # Write entry to disk
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                with open(dest_file_path, 'wb') as dest_file:
                    dest_file.write(zip_ref.read(src_file_path))
                
                # Validate destination file
                if os.path.getsize(dest_file_path) > 0:
                    success = True
                    break
                directory_move_log.info(f'Failed to write {dest_file_path}')
            except Exception as e:
                error_log.info(f'Error writing {dest_file_path}: {e}')

        # Handle severe fails
        if not success:
            error_log.info(f'Something is wrong with destination filesystem, panic: {dest_file_path}')
            raise Exception('Something is wrong with destination filesystem, panic')
        
        directory_move_log.info(f"Success: {dest_file_path}")
        
def organize_files(valid_entries, analyte_zip_found_log, directory_move_log, error_log, zip_file_path, base_dest_dir):
    """
    Organizes files by extracting and moving folders from a zip file based on valid entries.

    :param valid_entries: List of valid entries to process.
    :param analyte_zip_found_log: Logger for analyte zip found messages.
    :param directory_move_log: Logger for directory move messages.
    :param error_log: Logger for error messages.
    :param zip_file_path: Path to the zip file.
    :param base_dest_dir: Base destination directory.
    :return: DataFrame with species counts.
    """
    species_count = {}
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for genus, species, extern_id, target_position in valid_entries:
            # Find the corresponding folder in the zip file
            folder_name = None
            for file in zip_ref.namelist():
                if f'/{extern_id}/0_{target_position}/' in file:
                    folder_name = file
                    if folder_name != f'{file.split(extern_id)[0]}{extern_id}/0_{target_position}/':
                        raise Exception(f'Illegal state for {file}') # There is no spectrum for the extern_id but there is some folder with the extern_id's name
                    break
            
            if not folder_name:
                analyte_zip_found_log.info(f"❌ {extern_id} NOT FOUND\n")
                continue

            else:
                dest_dir = os.path.join(base_dest_dir, genus, species, extern_id[:8], f'0_{target_position}')
                zip_and_move_folder(zip_ref, folder_name, dest_dir, directory_move_log, error_log)

                # Update species count within genus
                if genus in species_count:
                    if species in species_count[genus]:
                        species_count[genus][species] += 1
                    else:
                        species_count[genus][species] = 1
                else:
                    species_count[genus] = {species: 1}

    # Convert species count to DataFrame
    data = []
    for genus, species_dict in species_count.items():
        for species, count in species_dict.items():
            data.append({'Genus': genus, 'Species': species, 'Count': count})
    df = pd.DataFrame(data, columns=['Genus', 'Species', 'Count'])
    return df
    
def find_xml_files(xml_found_log, analyte_validation_log, error_log, xml_file_path, year, test_mode=False):
    """
    Finds and processes XML files in a zip file within a specific year folder.

    :param xml_found_log: Logger for XML found messages.
    :param analyte_validation_log: Logger for analyte validation messages.
    :param error_log: Logger for error messages.
    :param xml_file_path: Path to the XML zip file.
    :param year: The year folder to look for XML files.
    :param test_mode: Boolean indicating if test mode is enabled.
    :return: List of valid entries.
    """
    valid_entries = []
    try:
        with zipfile.ZipFile(xml_file_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.startswith(f'{year}/') and file.endswith('.xml'):
                    xml_found_log.info(f"✅ {file} found\n")
                    with zip_ref.open(file) as f:
                        try:
                            tree = ET.parse(f)
                            root = tree.getroot()
                            for analyte in root.findall(".//Analyte"):
                                genus, species, extern_id, target_position = process_analyte(analyte, analyte_validation_log)
                                if genus and species and extern_id and target_position:
                                    valid_entries.append((genus, species.capitalize(), extern_id, target_position))
                                    if test_mode and len(valid_entries) >= 50:
                                        return valid_entries
                        except ET.ParseError as e:
                            error_log.info(f"Error parsing XML file {file}: {e}\n")
            if not valid_entries:
                xml_found_log.info(f"❌ No XML file found in {xml_file_path} for year {year}\n")
                raise Exception(f'No XML file found in the zip file for year {year}')
    except Exception as e:
        error_log.info(f"Error finding XML files: {e}\n")
    return valid_entries

def generate_reports(df, year):
    """
    Generates CSV, JSON, and XML reports from a DataFrame.

    :param df: DataFrame with species counts.
    :param year: Year for the report.
    :return: Paths to the generated CSV, JSON, and XML files.
    """
    stats_dir = f'/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/{year}/stats'
    os.makedirs(stats_dir, exist_ok=True)
    
    # Generate CSV
    csv_path = os.path.join(stats_dir, 'report.csv')
    df.to_csv(csv_path, index=False)
    
    # Generate JSON
    json_path = os.path.join(stats_dir, 'report.json')
    json_data = df.groupby('Genus', group_keys=False).apply(lambda x: x.set_index('Species')['Count'].to_dict()).to_dict()
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    
    
    # Generate XML
    xml_path = os.path.join(stats_dir, 'report.xml')
    root = ET.Element("Results")
    for genus, species_data in json_data.items():
        genus_element = ET.SubElement(root, "Genus", name=genus)
        for species, count in species_data.items():
            species_element = ET.SubElement(genus_element, "Species", name=species)
            species_element.text = str(count)
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    
    


def main():
    """
    Main function to run the data cleaner.
    """
    # Ask if this is a test run
    test_mode = input("Are you running a test? (yes/no): ").strip().lower() == 'yes'
    
    # Ask for the year to process
    year = input("Please enter the year you want to clean (e.g., 2020): ").strip()
    
    # Setup logging
    xml_found_log, analyte_validation_log, analyte_zip_found_log, directory_move_log, error_log = setup_logging(year)
    
    # Define paths
    base_dest_dir = f'/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/{year}/matched_bacteria/'
    xml_file_path = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/XML_2023.zip'
    zip_file_path = f'/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/{year}.zip'
    
    # Find and process XML files
    valid_entries = find_xml_files(xml_found_log, analyte_validation_log, error_log, xml_file_path, year, test_mode)
    
    # Organize files based on valid entries
    df = organize_files(valid_entries, analyte_zip_found_log, directory_move_log, error_log, zip_file_path, base_dest_dir)
    # Write the DataFrame to a text file
    txt_path = os.path.join(base_dest_dir, 'species_count.txt')
    with open(txt_path, 'w') as txt_file:
        txt_file.write(df.to_string(index=False))
    # Generate reports
    try:
        generate_reports(df, year)
        logging.info("Reports generated successfully.")
    except Exception as e:
        error_log.info(f"Error generating reports: {e}")

if __name__ == "__main__":
    main()