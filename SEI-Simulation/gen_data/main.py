import os
from utils_dataset_creation import get_files, assign_classes, surface_orientations, split_dataset, \
    save_dictionary, suppress_stdout_stderr, create_dataset_subprocess
from tqdm import tqdm

def main():
    # def generate_and_process_datasets():
    quantity = 1
    configfolder = 'SEI'
    configfile = 'config_SEI_HRTEM.json'

    rotation = 90

    application_path = os.path.join(os.getcwd(), '..', 'applications', configfolder)
    json_path = os.path.join(application_path, 'json_templates', configfile)
    structures_path = os.path.join(application_path, 'ATK_structures')

    xyz_files = get_files(structures_path)
    # xyz_paths = os.path.join(structures_path)
    class_dict = assign_classes(xyz_files)
    surf_orientations = surface_orientations(xyz_files)

    # Folder path
    folder_path = os.path.join(os.getcwd(), '..', '..', 'datagen')
    os.makedirs(folder_path, exist_ok=True)

    # File path
    file_path = os.path.join(folder_path, 'labels.json')
    save_dictionary(file_path, class_dict, indent=2)

    # print(f'Dictionary exported to {file_path}')



    for i in tqdm(xyz_files, desc='Processing files', unit='file', dynamic_ncols=True, leave=True):
        with suppress_stdout_stderr():
            structure = os.path.join(structures_path, i)
            create_dataset_subprocess(folder_path, quantity, rotation, structure, json_path, class_dict, surf_orientations)

    ML_dataset_path = os.path.join(os.getcwd(), '..', '..', 'Dataset')
    split_dataset(folder_path=folder_path, dataset_path=ML_dataset_path)


if __name__ == "__main__":
    main()