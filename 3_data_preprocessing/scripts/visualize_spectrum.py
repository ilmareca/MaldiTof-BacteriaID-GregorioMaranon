import matplotlib.pyplot as plt
from spectrum import SpectrumObject, Normalizer, VarStabilizer, Smoother, BaselineCorrecter, Trimmer
import os

# Visualización del espectro para cada paso
def visualize_step(spectrum, ax, title):
    ax.plot(spectrum.mz, spectrum.intensity)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('m/z', fontsize=8)
    ax.set_ylabel('Intensity', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)

# Procesamiento del espectro y retorno de los resultados de cada paso
def process_and_collect_steps(spectrum):
    steps = []

    # Step 0: Original Spectrum
    steps.append((spectrum, 'Original Spectrum'))

    # Step 1: Variance stabilization
    stabilizer = VarStabilizer(method="sqrt")
    spectrum = stabilizer(spectrum)
    steps.append((spectrum, 'Variance Stabilized'))

    # Step 2: Smoothing
    smoother = Smoother(halfwindow=10, polyorder=3)
    spectrum = smoother(spectrum)
    steps.append((spectrum, 'Smoothed'))

    # Step 3: Baseline correction
    baseline_correcter = BaselineCorrecter(method="SNIP", snip_n_iter=10)
    spectrum = baseline_correcter(spectrum)
    steps.append((spectrum, 'Baseline Corrected'))

    # Step 4: Normalization
    normalizer = Normalizer()
    spectrum = normalizer(spectrum)
    steps.append((spectrum, 'Normalized'))

    # Step 5: Trimming
    trimmer = Trimmer(min=2000, max=20000)
    spectrum = trimmer(spectrum)
    steps.append((spectrum, 'Trimmed'))

    return steps

# Crear la matriz de imágenes
def create_image_matrix(species_data, output_dir, output_filename):
    num_species = len(species_data)
    num_steps = 6  # Original + 5 steps de preprocesado

    fig, axes = plt.subplots(num_species, num_steps, figsize=(20, 4 * num_species), constrained_layout=True)
    if num_species == 1:
        axes = [axes]  # Asegurarse de que sea iterable si hay una sola especie

    for i, (species, files) in enumerate(species_data.items()):
        # Usar el primer espectro de cada especie
        acqu_file = files['acqu_files'][0]
        fid_file = files['fid_files'][0]
        spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)

        # Procesar el espectro y obtener los pasos
        steps = process_and_collect_steps(spectrum)

        # Dibujar cada paso en la fila correspondiente
        for j, (step_spectrum, step_title) in enumerate(steps):
            visualize_step(step_spectrum, axes[i][j], f'{species}\n{step_title}')

    # Guardar la matriz de imágenes
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Matriz de imágenes guardada en: {output_path}')

if __name__ == "__main__":
    # Definición de los datos de entrada
    species_data = {
        'Escherichia_Coli': {
            'acqu_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18252920/0_D5/1/1SLin/acqu'
            ],
            'fid_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Escherichia/Coli/18252920/0_D5/1/1SLin/fid'
            ]
        },
        'Enterococcus_Faecalis': {
            'acqu_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18128679/0_D2/1/1SLin/acqu'
            ],
            'fid_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Enterococcus/Faecalis/18128679/0_D2/1/1SLin/fid'
            ]
        },
        'Staphylococcus_Aureus': {
            'acqu_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18215596/0_B5/1/1SLin/acqu'
            ],
            'fid_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Aureus/18215596/0_B5/1/1SLin/fid'
            ]
        },
        'Klebsiella_Pneumoniaee': {
            'acqu_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/16061980/0_E5/1/1SLin/acqu'
            ],
            'fid_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Klebsiella/Pneumoniae/16061980/0_E5/1/1SLin/fid'
            ]
        },
        'Staphylococcus_Epidermidis': {
            'acqu_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18115317/0_D4/1/1SLin/acqu'
            ],
            'fid_files': [
                '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/2018/matched_bacteria/Staphylococcus/Epidermidis/18115317/0_D4/1/1SLin/fid'
            ]
        }
    }

    # Directorio de salida
    output_dir = '/export/usuarios01/ilmareca/github/MaldiTof-BacteriaID-GregorioMaranon/3_data_preprocessing/figures'
    output_filename = 'preprocessing_matrix.png'

    # Crear la matriz de imágenes
    create_image_matrix(species_data, output_dir, output_filename)
