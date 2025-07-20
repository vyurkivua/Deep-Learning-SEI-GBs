import json
from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.patches import Polygon
from xml.etree.ElementTree import Element, SubElement, tostring
from _misc_utils import find_components_from_dir
from xml.dom import minidom
import cv2
from _misc_utils import extract_points_from_polygon

def round_dict_values(dictionary, decimals=2):
    """
    Recursively round all numeric values in a dictionary to the specified number of decimals.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            round_dict_values(value, decimals)
        elif isinstance(value, (float, np.float64)):
            dictionary[key] = round(value, decimals)
        elif isinstance(value, list):
            for i in range(len(value)):
                if type(value[i]) == float:
                    value[i] = round(value[i], decimals)
            # round_dict_values(value, decimals)


def save_dictionary(file_path, dictionary, indent=2):
    # Round off all numeric values in the dictionary to 2 decimals
    round_dict_values(dictionary, decimals=2)

    # Write the dictionary to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=indent)


# def save_dictionary(config, stem_img, hrtem_img, npy_file):
#     struct = {'STEM': stem_img}
#     struct['HRTEM'] = hrtem_img
#
#     save_path = '{}'.format(npy_file)
#     print(f'Saving simulated image and CHs in {save_path}')
#
#     np.save(save_path, struct)
#
#     # struct = np.load(npy_file, allow_pickle=True).item()

def save_tem_image(tem_img, saving_name):
    fig, axs = plt.subplots(1, 1)
    ax_stem = axs
    im_stem = ax_stem.imshow(tem_img, cmap='gray')
    ax_stem.axis('off')
    plt.savefig(saving_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    image = Image.open(saving_name)
    resized_image = image.resize((256, 256), Image.Resampling.LANCZOS)
    resized_image.save(saving_name, format='PNG')

    # plt.show()
    # plt.close()


def save_atomic_structure(rotated_image, atomic_structure):
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.axis('off')
    axes.imshow(rotated_image, cmap='gray')
    axes.set_title('Atomic Structure')

    # Save the rotated image with the desired filename
    annotated_image_with_atoms = atomic_structure
    plt.savefig(annotated_image_with_atoms)
    # plt.show()
    plt.close(fig)




def save_npy_file(config, stem_img, hrtem_img,npy_file):
    struct = {'STEM': stem_img}
    struct['HRTEM'] = hrtem_img
    # make_folder(config.data_kwargs.data_folder_path)
    #
    # save_path = os.path.join(config.data_kwargs.data_folder_path,
    #                          '{}.npy'.format(npy_file))
    save_path = '{}'.format(npy_file)
    # print(f'Saving simulated image and CHs in {save_path}')

    np.save(save_path, struct)

    struct = np.load(npy_file, allow_pickle=True).item()


def plot_image_with_atoms(hrtem_img, annotated_image_with_atoms, model, atomic_structure, params_dict):
    fig, axes = plt.subplots(1, 4, figsize=(18, 8), width_ratios=[1, 1, 1, 0.3])

    # Plot Atomic Structure
    model.write('atomic_structure.png', format='png')
    atomic_structure_img = plt.imread('atomic_structure.png')
    rotated_image = np.ascontiguousarray(np.rot90(atomic_structure_img, k=3))
    axes[0].axis('off')
    axes[0].imshow(rotated_image, cmap='gray')
    axes[0].set_title('Atomic Structure')

    # Plot HRTEM Image
    axes[1].imshow(hrtem_img, cmap='gray')
    axes[1].set_title('STEM Image')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')

    # Plot Experimental Image
    img = cv2.imread('Experimental_image.png')  # np.asarray(Image.open('Experimental_image.png'))
    axes[2].imshow(img)
    axes[2].set_title('Experimental Image')
    axes[2].axis('off')

    # Show params_dict at the last image axis
    axes[3].axis('off')
    params_str = '\n'.join([f'{key}: {value}' for key, value in params_dict.items()])

    # Calculate the position of text dynamically based on the size of the axis
    text_x = 0.05  # Adjust the x-coordinate as needed
    text_y = 0.8  # Adjust the y-coordinate as needed
    axes[3].text(text_x, text_y, params_str, ha='left', va='top', fontsize=10, wrap=True)

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)  # Adjust left and right margins

    plt.tight_layout(pad=0.5)
    plt.savefig(annotated_image_with_atoms)
    # plt.show()
    # plt.close()
    plt.close(fig)

    # Save Atomic Structure
    save_atomic_structure(rotated_image, atomic_structure)
    os.remove('atomic_structure.png')



def plot_hrtem_image_with_atoms(hrtem_img, annotated_hrtem_image_with_atoms, model, atomic_structure):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Plot Atomic Structure
    model.write('atomic_structure.png', format='png')
    atomic_structure_img = plt.imread('atomic_structure.png')
    rotated_image = np.ascontiguousarray(np.rot90(atomic_structure_img, k=3))
    axes[0].axis('off')
    axes[0].imshow(rotated_image, cmap='gray')
    axes[0].set_title('Atomic Structure')

    # Plot HRTEM Image
    axes[1].imshow(hrtem_img, cmap='gray')
    axes[1].set_title('STEM Image')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')
    axes[1].axis('off')

    plt.tight_layout(pad=0.5)
    plt.savefig(annotated_hrtem_image_with_atoms)
    # plt.show()
    # plt.close()
    plt.close(fig)

    # Save Atomic Structure
    save_atomic_structure(rotated_image, atomic_structure)
    os.remove('atomic_structure.png')


def plot_clusters(data_XY_new, grain_boundary_indices_list, left_grain_indices_list,
                  right_grain_indices_list, hrtem_img, xyz_path, annotated_image_with_atoms, model, atomic_structure):
    c1, c2 = find_components_from_dir(xyz_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Plot XY left grain points
    axes[1].scatter(data_XY_new[left_grain_indices_list, 0], data_XY_new[left_grain_indices_list, 1],
                    color='yellow', label='{}'.format(c1))

    # Plot XY right grain points
    axes[1].scatter(data_XY_new[right_grain_indices_list, 0], data_XY_new[right_grain_indices_list, 1],
                    color='purple', label='{}'.format(c2))

    # Plot XY grain boundary
    axes[1].scatter(data_XY_new[grain_boundary_indices_list, 0], data_XY_new[grain_boundary_indices_list, 1],
                    color='red', label='Grain Boundary')

    axes[1].imshow(hrtem_img, cmap='gray')
    axes[1].set_title('Clustered Stem Image ')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')
    axes[1].legend()

    # Calculate the convex hull
    hull_right = ConvexHull(data_XY_new[right_grain_indices_list])
    hull_left = ConvexHull(data_XY_new[left_grain_indices_list])
    hull_mid = ConvexHull(data_XY_new[grain_boundary_indices_list])

    # Draw polygons using the hull points
    polygon_left = Polygon(data_XY_new[left_grain_indices_list][hull_left.vertices], edgecolor='yellow',
                           facecolor='none', label='{}'.format(c1))
    polygon_right = Polygon(data_XY_new[right_grain_indices_list][hull_right.vertices], edgecolor='purple',
                            facecolor='none', label='{}'.format(c2))
    polygon_mid = Polygon(data_XY_new[grain_boundary_indices_list][hull_mid.vertices], edgecolor='red',
                          facecolor='none', label='Grain Boundary')

    # Add polygons to the plot
    axes[2].add_patch(polygon_right)
    axes[2].add_patch(polygon_left)
    axes[2].add_patch(polygon_mid)
    axes[2].imshow(hrtem_img, cmap='gray')
    # axes[2].imshow(stem_img, cmap='gray')
    axes[2].set_title('Labeled Stem Image')
    axes[2].set_xlabel('X-axis')
    axes[2].set_ylabel('Y-axis')
    axes[2].legend()

    model.write('atomic_structure.png', format='png')

    # print(dir(model))
    #
    # print(model.rotate())

    atomic_structure_img = plt.imread('atomic_structure.png')

    rotated_image = np.ascontiguousarray(np.rot90(atomic_structure_img, k=3))

    axes[0].axis('off')
    axes[0].imshow(rotated_image, cmap='gray')

    # ###########
    # axes[0].axis('off')
    # axes[0].imshow(atomic_structure_img, cmap='gray')
    axes[0].set_title('Atomic Structure')
    # axes[0].set_xlabel('X-axis')
    # axes[0].set_ylabel('Y-axis')

    plt.savefig(annotated_image_with_atoms)

    # Close the plots
    # plt.close()
    # plt.show()
    plt.close(fig)

    # Save Atomic Structure
    save_atomic_structure(rotated_image, atomic_structure)

    os.remove('atomic_structure.png')
    # plt.show()
    # plt.close()

    return polygon_right, polygon_left, polygon_mid



def save_image_with_unique_name(directory, img, filename_base, cmap='gray'):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # List existing files in the directory
    existing_files = os.listdir(directory)

    # Find the highest index of files with the same base name
    max_index = -1
    for file in existing_files:
        if file.startswith(filename_base) and file.endswith('.png'):
            try:
                index = int(file[len(filename_base) + 1:file.rindex('.png')])
                if index > max_index:
                    max_index = index
            except ValueError:
                continue

    # Determine the new file name
    new_filename = f"{filename_base}_{max_index + 1}.png"
    new_filepath = os.path.join(directory, new_filename)

    # Save the image
    plt.imsave(new_filepath, img, cmap=cmap)
    # print(f"Image saved as {new_filepath}")



def find_annotation_points(polygon_right, polygon_left, polygon_mid, stem_img, saved_name_stem, xyz_path):
    # c1, c2 = find_components(xyz_path)
    c1, c2 = find_components_from_dir(xyz_path)
    points_right = extract_points_from_polygon(polygon_right)
    points_left = extract_points_from_polygon(polygon_left)
    points_mid = extract_points_from_polygon(polygon_mid)

    # Create objects list with points
    objects_list = [
        {
            "name": c1,
            "pose": "example_pose",
            "truncated": 0,
            "difficult": 0,
            "points": points_left,
            "xmin": min(p[0] for p in points_left),
            "ymin": min(p[1] for p in points_left),
            "xmax": max(p[0] for p in points_left),
            "ymax": max(p[1] for p in points_left)
        },

        {
            "name": c2,
            "pose": "example_pose",
            "truncated": 0,
            "difficult": 0,
            "points": points_right,
            "xmin": min(p[0] for p in points_right),
            "ymin": min(p[1] for p in points_right),
            "xmax": max(p[0] for p in points_right),
            "ymax": max(p[1] for p in points_right)
        },

        {
            "name": "Grain_boundary",
            "pose": "example_pose",
            "truncated": 0,
            "difficult": 0,
            "points": points_mid,
            "xmin": min(p[0] for p in points_mid),
            "ymin": min(p[1] for p in points_mid),
            "xmax": max(p[0] for p in points_mid),
            "ymax": max(p[1] for p in points_mid)
        }
    ]

    # Example annotation parameters
    annotation_params = {
        "folder": "/",
        "filename": saved_name_stem,  # "stem_image.png",
        "path": "",
        "database": "",
        "width": stem_img.shape[0],
        "height": stem_img.shape[1],
        "depth": 3,
        "segmented": 0,
        "objects": objects_list
    }
    return annotation_params





# Function to create an annotation XML string
def create_annotation_xml(xmlname, folder, filename, path, database, width, height, depth, segmented, objects):
    annotation = Element("annotation")

    folder_elem = SubElement(annotation, "folder")
    folder_elem.text = folder

    filename_elem = SubElement(annotation, "filename")
    filename_elem.text = filename

    path_elem = SubElement(annotation, "path")
    path_elem.text = path

    source_elem = SubElement(annotation, "source")
    database_elem = SubElement(source_elem, "database")
    database_elem.text = database

    size_elem = SubElement(annotation, "size")
    width_elem = SubElement(size_elem, "width")
    width_elem.text = str(width)
    height_elem = SubElement(size_elem, "height")
    height_elem.text = str(height)
    depth_elem = SubElement(size_elem, "depth")
    depth_elem.text = str(depth)

    segmented_elem = SubElement(annotation, "segmented")
    segmented_elem.text = str(segmented)

    for obj in objects:
        object_elem = SubElement(annotation, "object")

        name_elem = SubElement(object_elem, "name")
        name_elem.text = obj["name"]

        pose_elem = SubElement(object_elem, "pose")
        pose_elem.text = obj["pose"]

        truncated_elem = SubElement(object_elem, "truncated")
        truncated_elem.text = str(obj["truncated"])

        difficult_elem = SubElement(object_elem, "difficult")
        difficult_elem.text = str(obj["difficult"])

        if "points" in obj:
            polygon_elem = SubElement(object_elem, "polygon")
            for i, point in enumerate(obj["points"]):
                x_elem = SubElement(polygon_elem, f"x{i + 1}")
                x_elem.text = str(point[0])
                y_elem = SubElement(polygon_elem, f"y{i + 1}")
                y_elem.text = str(point[1])

        bndbox_elem = SubElement(object_elem, "bndbox")
        xmin_elem = SubElement(bndbox_elem, "xmin")
        xmin_elem.text = str(obj["xmin"])
        ymin_elem = SubElement(bndbox_elem, "ymin")
        ymin_elem.text = str(obj["ymin"])
        xmax_elem = SubElement(bndbox_elem, "xmax")
        xmax_elem.text = str(obj["xmax"])
        ymax_elem = SubElement(bndbox_elem, "ymax")
        ymax_elem.text = str(obj["ymax"])

    xml_string = tostring(annotation, encoding="utf-8").decode("utf-8")
    xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="  ")
    # return xml_pretty

    with open(xmlname, "w") as xml_file:
        xml_file.write(xml_pretty)




def generate_mask_from_annotations(polygon_right, polygon_left, polygon_mid, filename, target_size, xyz_path,
                                   class_dict, npy_file_labels, consider_miller_indices):
    c1, c2 = find_components_from_dir(xyz_path)

    points_right = extract_points_from_polygon(polygon_right)
    points_left = extract_points_from_polygon(polygon_left)
    points_mid = extract_points_from_polygon(polygon_mid)

    classes = ['Background', c1, c2, 'Grain Boundary']


    class_mapping = class_dict

    # Initialize an empty RGB mask
    # mask = np.zeros((*target_size, 3), dtype=np.uint8)
    mask = np.zeros(target_size, dtype=np.uint8)

    for region in classes:  # Include 'Background'
        if region == c1:
            points = points_left
        elif region == c2:
            points = points_right
        elif region == 'Grain Boundary':
            points = points_mid
        else:
            continue

        # Fill the polygon with the class color
        polygon_array = np.array(points, dtype=np.int32)
        class_color = class_mapping[region]
        cv2.fillPoly(mask, [polygon_array], class_color)

    Image.fromarray(mask).save(filename)
    np.save(npy_file_labels, mask)
    # print('Saving to', filename)
    return mask

