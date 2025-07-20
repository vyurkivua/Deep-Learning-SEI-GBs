import os
# import tensorflow as tf
import sys

# os.environ["OMP_NUM_THREADS"] = '1'
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# sys.path.insert(0, os.path.abspath('../'))
import numpy as np
import random

sys.path.insert(0, os.path.abspath('../'))
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import shutil
import gc  # Import the garbage collector
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.patches import Polygon
import warnings
# from sklearn.cluster import KMeans
from multiprocessing import Process

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
import json
from tqdm import tqdm
from _simulate_image_utils import simulate_tem_image
from _save_files_utils import save_dictionary,save_dictionary,save_tem_image,save_atomic_structure,save_npy_file,\
    plot_image_with_atoms, plot_hrtem_image_with_atoms, plot_clusters, save_image_with_unique_name, \
    find_annotation_points, create_annotation_xml, generate_mask_from_annotations
from _misc_utils import test_scaling_factor, save_pixel_intensity, find_components_from_dir

from PIL import Image, ImageDraw
import os
import numpy as np
import cv2


# from Sensitivity_analysis import simulate_tem_image




def visualize_results(stem_img, hrtem_img):
    fig = plt.figure(figsize=(15, 8))

    ax = fig.add_subplot(1, 2, 1)
    im = ax.imshow(stem_img, cmap='gray')
    plt.title('STEM Image IFFT', fontsize=20)
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax1)

    ax = fig.add_subplot(1, 2, 2)
    im = ax.imshow(hrtem_img, cmap='gray')
    plt.title('STEM Image', fontsize=20)
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax1)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def find_dataset_XY(scaling_factor_x_stem, coordinates):
    scaling_factor_y_stem = scaling_factor_x_stem
    X_scaled = coordinates[:, 1] * scaling_factor_x_stem
    Y_scaled = coordinates[:, 0] * scaling_factor_y_stem

    # Create data arrays by stacking X and Y, and X and Z
    data_XY_scaled = np.column_stack(
        (X_scaled, Y_scaled))  # np.column_stack((X_scaled, Y_scaled))  # Combine X and Y data
    return data_XY_scaled


def rotate_points(original_points, angle_degrees):
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), np.sin(angle_radians)],
                                [-np.sin(angle_radians), np.cos(angle_radians)]])

    data_XY_rotated = np.dot(original_points, rotation_matrix)
    return data_XY_rotated


def use_ML(data_XY):
    # Create and fit the KMeans model for XY data
    kmeans_model_XY = KMeans(n_clusters=2, init='k-means++',
                             n_init='auto')  # Adjust the number of clusters if necessary
    kmeans_model_XY.fit(data_XY)

    # Predict cluster labels for XY data
    labels_XY = kmeans_model_XY.predict(data_XY)

    # Calculate the midpoint between clusters as the XY grain boundary
    left_grain_max_x_XY = np.max(data_XY[labels_XY == 0, 0])
    right_grain_min_x_XY = np.min(data_XY[labels_XY == 1, 0])
    grain_boundary_x_XY = (left_grain_max_x_XY + right_grain_min_x_XY) / 2

    return grain_boundary_x_XY


# Use No ML
def use_no_ML(data_XY):
    # Create and fit the KMeans model for XY data
    left_grain_max_x_XY = np.max(data_XY[:, 0])
    right_grain_min_x_XY = np.min(data_XY[:, 0])
    grain_boundary_x_XY = (left_grain_max_x_XY + right_grain_min_x_XY) / 2
    #     grain_boundary_x_XY = np.mean(left_grain_max_x_XY+right_grain_min_x_XY)

    return grain_boundary_x_XY

def find_grains(data_XY, boundary_width_XY, grain_boundary_x_XY):
    # find_column_heights(data_XY)

    left_grain_indices_XY = np.where(data_XY[:, 0] < grain_boundary_x_XY - boundary_width_XY)[0]
    right_grain_indices_XY = np.where(data_XY[:, 0] > grain_boundary_x_XY + boundary_width_XY)[0]
    # boundary_grain_indices_XY = np.where(data_XY[:, 0] == grain_boundary_list)

    right_x_list = []
    left_x_list = []
    for i in data_XY[:, 0]:
        if i < grain_boundary_x_XY - boundary_width_XY:
            left_x_list.append(i)

        elif i > grain_boundary_x_XY + boundary_width_XY:
            right_x_list.append(i)

    grain_boundary_indices_list = []
    j = 0

    for i in data_XY[:, 0]:
        if max(left_x_list)-.001 <=  i <= min(right_x_list)+.001:
            grain_boundary_indices_list.append(j)

        j+=1

    left_grain_indices_list = list(left_grain_indices_XY)
    right_grain_indices_list = list(right_grain_indices_XY)

    return grain_boundary_indices_list, left_grain_indices_list, right_grain_indices_list


def get_surface_orientation(xyz_file):
    component_list = ['LiF', 'Li2O', 'Li2CO3', 'LiOH']

    component_1, component_2 = find_components_from_dir(xyz_file, surface_orientation=True)

    for i in component_list:
        if i in component_1:
            mill_idx1 = component_1.removeprefix(i)

        if i in component_2:
            mill_idx2 = component_2.removeprefix(i)

    return mill_idx1, mill_idx2



def create_dataset_surf_or(xyz_path, stem_img, polygon_right, polygon_left, polygon_mid, output_dir):
    # c1, c2 = find_components_from_dir(xyz_path, surface_orientation= True)

    mill_idx1, mill_idx2 = get_surface_orientation(xyz_path)

    # Convert polygons to Path objects for easy checking
    path_right = Path(polygon_right.get_xy())
    path_left = Path(polygon_left.get_xy())
    path_mid = Path(polygon_mid.get_xy())

    # Get image dimensions
    height, width = stem_img.shape

    # Create blank images with black background
    img_right = np.zeros_like(stem_img)
    img_left = np.zeros_like(stem_img)
    img_mid = np.zeros_like(stem_img)

    # Create a mask for each polygon
    for y in range(height):
        for x in range(width):
            if path_right.contains_point((x, y)):
                img_right[y, x] = stem_img[y, x]
            if path_left.contains_point((x, y)):
                img_left[y, x] = stem_img[y, x]
            if path_mid.contains_point((x, y)):
                img_mid[y, x] = stem_img[y, x]

    correct_output_dir = os.path.dirname(output_dir)
    filenames = mill_idx1, mill_idx2
    # print(output_dir)
    # print(filename)


    output_dir = os.path.join(correct_output_dir, filenames[0])
    save_image_with_unique_name(output_dir, img_left, filenames[0])
    output_dir = os.path.join(correct_output_dir, filenames[1])
    save_image_with_unique_name(output_dir, img_right, filenames[1])



def plot_on_stem(data_XY_scaled, stem_img, polygon_right, polygon_left, polygon_mid, annotated_image, xyz_path):
    c1, c2 = find_components_from_dir(xyz_path)

    # Create a figure and axis for the STEM Image
    fig_stem, ax_stem = plt.subplots(1, 1)
    X_scaled, Y_scaled = data_XY_scaled[:, 0], data_XY_scaled[:, 1]

    # Plot the STEM Image
    im_stem = ax_stem.imshow(stem_img, cmap='gray')
    ax_stem.set_title('Inverse FFT STEM Image', fontsize=20)

    # Add colorbar
    divider_stem = make_axes_locatable(ax_stem)
    cax_stem = divider_stem.append_axes("right", size="5%", pad=0.05)
    cbar_stem = plt.colorbar(im_stem, cax=cax_stem)

    # Plot bounding boxes for XY grains and grain boundary on the stem image
    # ax_stem.scatter(X_scaled, Y_scaled, c='red', marker='o', label='Coordinates')

    # Add polygons with labels
    ax_stem.add_patch(Polygon(polygon_right.get_xy(), edgecolor='green', facecolor='none', label=c2))
    ax_stem.add_patch(Polygon(polygon_left.get_xy(), edgecolor='blue', facecolor='none', label=c1))
    ax_stem.add_patch(Polygon(polygon_mid.get_xy(), edgecolor='red', facecolor='none', label='Grain Boundary'))

    # Display legend
    ax_stem.legend()

    plt.savefig(annotated_image)

    # plt.show()
    # Save the annotated image

    # Close the figures to avoid memory leaks
    plt.close()
    plt.close(fig_stem)


def plot_clusters_on_stem_image(boundary_width_XY, data_XY_scaled, rotation_angle, stem_img, hrtem_img, xyz_path,
                                annotated_image_with_atoms, structure_img_exp_specs_img,
                                annotated_hrtem_image_with_atoms, model, atomic_structure, params_dic):
    # boundary_width_XY = 3* scaling_factor_x_stem
    data_XY_unrotated = rotate_points(data_XY_scaled, 270 + rotation_angle)  # np.arccos(rotation_angle))
    theta = rotation_angle

    alpha = (360 - theta) % 360

    data_XY_new = rotate_points(data_XY_unrotated, 90 + alpha)
    # grain_boundary_x_XY = use_ML(data_XY_unrotated)
    grain_boundary_x_XY = use_no_ML(data_XY_unrotated)
    grain_boundary_indices_list, left_grain_indices_list, right_grain_indices_list = find_grains(data_XY_unrotated,
                                                                                                 boundary_width_XY,
                                                                                                 grain_boundary_x_XY)

    polygon_right, polygon_left, polygon_mid = plot_clusters(data_XY_new, grain_boundary_indices_list,
                                                             left_grain_indices_list, right_grain_indices_list,
                                                             hrtem_img, xyz_path, annotated_image_with_atoms, model,
                                                             atomic_structure)

    plot_image_with_atoms(hrtem_img, structure_img_exp_specs_img, model, atomic_structure, params_dic)
    plot_hrtem_image_with_atoms(hrtem_img, annotated_hrtem_image_with_atoms, model, atomic_structure)

    return data_XY_new, polygon_right, polygon_left, polygon_mid


def get_files(structures_path):
    xyz_files = []
    for root, dirs, files in os.walk(structures_path):
        for file in files:
            if file.endswith('.xyz'):  # Assuming .xyz files
                xyz_files.append(os.path.join(root, file))
                # xyz_files.append(file)

    print('Total structures', len(xyz_files))
    return xyz_files




#
def find_components(xyz_path):
    filename_parts = xyz_path[:-12].split("_")
    c1, c2 = filename_parts[0], filename_parts[1]

    # def find_components(xyz_path):
    #     filename_parts = xyz_path.split("\\")[-1][:-12].split("_")
    #     c1, c2 = filename_parts[0], filename_parts[1]

    return c1, c2





def copy_files_to_ml_model(src, dst):
    shutil.copytree(src, dst)


def create_directories(*paths):
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def input_model(rotation, xyz_path, JSON_PATH):
    config = get_config(JSON_PATH)
    # xyz_path = config.config_dict['data_kwargs']['xyz_path']
    # xyz_path = os.path.join(script_dir, xyz_path)
    current_model = ATK_Random_HEA(path=xyz_path,
                                   spatial_domain=(config.structure_kwargs.spatial_domain,),
                                   transl_xy=None,
                                   rot_y=[rotation])

    model, coordinates, rotation_angle = current_model.get_model()
    # model, coordinates, rotation_angle = current_model.get_model()

    model = current_model.model
    # view(model)
    return config, model, coordinates, rotation_angle



def assign_classes(xyz_files):
    components_mapping = {}

    for xyz_file in xyz_files:
        component_1, component_2 = find_components_from_dir(xyz_file, surface_orientation = False)

        if component_1 not in components_mapping:
            components_mapping[component_1] = len(components_mapping) + 1

        if component_2 not in components_mapping:
            components_mapping[component_2] = len(components_mapping) + 1

    gray_value = 2  # Starting grayscale value
    class_dict = dict()
    class_dict['Background'] = 0  # Grayscale value for Background
    class_dict['Grain Boundary'] = 1  # Grayscale value for Background

    for i in components_mapping.keys():
        if gray_value < 256:
            components_mapping[i] = gray_value
            gray_value += 1

        class_dict[i] = components_mapping[i]


    return class_dict



def surface_orientations(xyz_files):
    surf_orientations = []
    component_list = ['LiF', 'Li2O', 'Li2CO3', 'LiOH']
    for xyz_file in xyz_files:
        component_1, component_2 = find_components_from_dir(xyz_file, surface_orientation=True)

        for i in component_list:
            if i in component_1:
                mill_idx1 = component_1.removeprefix(i)
                surf_orientations.append(mill_idx1)

            if i in component_2:
                mill_idx2 = component_2.removeprefix(i)
                surf_orientations.append(mill_idx2)

    return surf_orientations


"""def assign_classes(xyz_files):
    components_mapping = {}

    for xyz_file in xyz_files:
        component_1, component_2 = find_components(xyz_file)

        if component_1 not in components_mapping:
            components_mapping[component_1] = len(components_mapping) + 1

        if component_2 not in components_mapping:
            components_mapping[component_2] = len(components_mapping) + 1

    labels = [3, 0, 0]
    class_dict = dict()
    class_dict['Background'] = (1,0,0)
    class_dict['Grain Boundary'] = (0,0,255)

    for i in components_mapping.keys():
        if labels[0] < 256:  # assign a color channel, starting from Red
            components_mapping[i] = tuple(labels)
            labels[0] += 1

        elif labels[1] < 256:  # assign a color channel, starting from Green
            components_mapping[i] = tuple(labels)
            labels[1] += 1

        elif labels[2] < 256:  # assign a color channel, starting from Blue
            components_mapping[i] = tuple(labels)
            labels[2] += 1

        class_dict[i] = components_mapping[i]

    return class_dict"""



def create_file_objects(datagen_path, base_filename, index):
    file_specs = {
        'stem_image': {'folder': 'images_stem_ifft', 'prefix': 'stem_image', 'extension': 'png'},
        'hrtem_image': {'folder': 'images_hrtem', 'prefix': 'hrtem_image', 'extension': 'png'},
        'annotations': {'folder': 'annotations', 'prefix': 'annotations', 'extension': 'xml'},
        'annotated_image': {'folder': 'annotated_images', 'prefix': 'annotated_images', 'extension': 'png'},
        'atomic_structure': {'folder': 'atomic_structures', 'prefix': 'atomic_structure', 'extension': 'png'},
        'annotated_image_with_atoms': {'folder': 'annotated_images_atoms', 'prefix': 'annotated_images_atoms', 'extension': 'png'},
        'structure_img_exp_specs_img': {'folder': 'image_with_specs', 'prefix': 'image_with_specs', 'extension': 'png'},
        'annotated_hrtem_image_with_atoms': {'folder': 'annotated_hrtem_image_with_atoms', 'prefix': 'annotated_images_atoms', 'extension': 'png'},
        'npy_file': {'folder': 'npy_files', 'prefix': 'npy_image', 'extension': 'npy'},
        'label_file_npy': {'folder': 'npy_labels', 'prefix': 'npy_label', 'extension': 'npy'},
        'label_file_png': {'folder': 'labels', 'prefix': 'label', 'extension': 'png'},
        'microscopic_params': {'folder': 'microscopic_parameters', 'prefix': 'parameters', 'extension': 'json'},
        'surface_orientation_dir': {'folder': 'surface_orientation', 'prefix': 'surface_orientation', 'extension': 'png'}
    }

    file_paths = {}

    for key, spec in file_specs.items():
        file_obj = File_object(
            datagen_path,
            spec['folder'],
            spec['prefix'],
            f'{base_filename}_{index}',
            spec['extension']
        )
        file_paths[key] = file_obj.make_folder()

    return file_paths

def save_simulated_files(config,stem_img, hrtem_img, file_paths,params_dict,scaling_factor_x_stem, coordinates,
                         boundary_width_XY, xyz_path, model, rotation_angle, class_dict):
    # Use the file paths as inputs to your functions
    save_npy_file(config, stem_img, hrtem_img, file_paths['npy_file'])

    save_tem_image(stem_img, file_paths['stem_image'])

    mean_pixel_int = save_pixel_intensity(file_paths['stem_image'])
    params_dict['Mean_pixel_intensity_STEM_IFFT'] = mean_pixel_int
    save_dictionary(file_paths['microscopic_params'], params_dict, indent=4)

    save_tem_image(hrtem_img, file_paths['hrtem_image'])

    data_XY_scaled = find_dataset_XY(scaling_factor_x_stem, coordinates)

    data_XY_new, polygon_right, polygon_left, polygon_mid = plot_clusters_on_stem_image(
        boundary_width_XY, data_XY_scaled, rotation_angle, stem_img, hrtem_img, xyz_path,
        file_paths['annotated_image_with_atoms'], file_paths['structure_img_exp_specs_img'],
        file_paths['annotated_hrtem_image_with_atoms'], model, file_paths['atomic_structure'], params_dict
    )

    plot_on_stem(data_XY_new, stem_img, polygon_right, polygon_left, polygon_mid, file_paths['annotated_image'],
                 xyz_path)

    create_dataset_surf_or(xyz_path, stem_img, polygon_right, polygon_left, polygon_mid,
                           file_paths['surface_orientation_dir'])
    annotation_params = find_annotation_points(polygon_right, polygon_left, polygon_mid, stem_img,
                                               file_paths['stem_image'], xyz_path)

    create_annotation_xml(file_paths['annotations'], **annotation_params)

    target_size = (config.TEM_image_kwargs.image_size,config.TEM_image_kwargs.image_size)

    mask = generate_mask_from_annotations(
        polygon_right, polygon_left, polygon_mid, file_paths['label_file_png'], target_size,
        xyz_path, class_dict, file_paths['label_file_npy'], consider_miller_indices=False
    )

def create_dataset(datagen_path, quantity, rotation, xyz_path, JSON_PATH, class_dict, surf_orientations):
    for i in range(0, quantity):
        base_filename = os.path.basename(xyz_path)[:-4]
        file_paths = create_file_objects(datagen_path, base_filename, i)
        config, model, coordinates, rotation_angle = input_model(rotation, xyz_path, JSON_PATH)
        spatial_domain = config.structure_kwargs.spatial_domain
        scaling_factor_x_stem = 258 / spatial_domain
        boundary_width_XY = 2 * scaling_factor_x_stem

        stem_img, hrtem_img, params_dict = simulate_tem_image(config, model)

        save_simulated_files(config, stem_img, hrtem_img, file_paths, params_dict, scaling_factor_x_stem, coordinates,
                             boundary_width_XY, xyz_path, model, rotation_angle, class_dict)



class File_object:
    def __init__(self, datagen_path, folder_name, file_prefix, filename, file_extension):
        self.datagen_path = datagen_path
        self.folder_name = folder_name
        self.file_prefix = file_prefix
        self.filename = filename
        self.file_extension = file_extension

    def make_folder(self):
        saved_name_stem = os.path.join(
            self.datagen_path, self.folder_name, f'{self.file_prefix}_{self.filename}.{self.file_extension}')
        os.makedirs(os.path.dirname(saved_name_stem), exist_ok=True)
        return saved_name_stem


def split_dataset(folder_path='../../datagen', dataset_path='Dataset', train_ratio=0.95, val_ratio=0, test_ratio=0.05):
    # Ensure the ratios sum up to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of train, validation, and test ratios must be 1."

    # Create directories for train, validation, and test sets
    os.makedirs(os.path.join(dataset_path, 'train', 'image'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'train', 'label'), exist_ok=True)
    if val_ratio>0:
        os.makedirs(os.path.join(dataset_path, 'validation', 'image'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'validation', 'label'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'test', 'image'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'test', 'label'), exist_ok=True)

    # Get list of files in images and labels directory
    images_path = os.path.join(folder_path, 'images_hrtem')
    labels_path = os.path.join(folder_path, 'labels')

    image_files = os.listdir(images_path)
    label_files = os.listdir(labels_path)

    # Strip the prefixes to match image and label files
    image_files_stripped = [file.replace('hrtem_image_', '') for file in image_files]
    label_files_stripped = [file.replace('label_', '') for file in label_files]


    # Ensure the stripped filenames match
    assert set(image_files_stripped) == set(label_files_stripped), "Image and label files do not match."

    # Shuffle the stripped file list
    files = list(image_files_stripped)
    random.shuffle(files)

    # Calculate split indices
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Split the files
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    def move_files(file_list, split):
        for file in tqdm(file_list, desc=f'Moving files to {split}', unit='file'):
            hrtem_image_file = 'hrtem_image_' + file
            label_file = 'label_' + file
            shutil.copyfile(os.path.join(images_path, hrtem_image_file),
                        os.path.join(dataset_path, split, 'image', hrtem_image_file))
            shutil.copyfile(os.path.join(labels_path, label_file), os.path.join(dataset_path, split, 'label', label_file))
        shutil.copyfile(os.path.join(folder_path, 'labels.json'),
                        os.path.join(dataset_path,  'labels.json'))

    # Move the files to respective directories
    move_files(train_files, 'train')
    move_files(val_files, 'validation')
    move_files(test_files, 'test')

    print(f'Train files: {len(train_files)}, Validation files: {len(val_files)}, Test files: {len(test_files)}')
    print('Dataset split completed.')


class suppress_stdout_stderr(object):
    '''
    A context manager to suppress stdout and stderr
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def create_dataset_subprocess(*args):
    """
    Wrapper function to run create_dataset in a separate process.
    """
    p = Process(target=create_dataset, args=args)
    p.start()
    p.join()