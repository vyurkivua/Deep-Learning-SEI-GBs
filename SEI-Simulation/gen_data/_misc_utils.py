import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xml.etree.ElementTree as ET
from matplotlib.patches import Polygon
import cv2
from PIL import Image, ImageDraw
import numpy as np
import os



def test_scaling_factor(coordinates, stem_img, hrtem_img, scaling_factor_x_stem):
    # for i in np.arange(8,9,.1):
    # scaling_factor_x_stem = 8.6
    scaling_factor_y_stem = scaling_factor_x_stem

    X_scaled = coordinates[:, 1] * scaling_factor_x_stem  # Initialize an empty list for X-coordinate data
    Y_scaled = coordinates[:, 0] * scaling_factor_y_stem  # Initialize an empty list for Y-coordinate data

    # Assuming 'coordinates' is a NumPy array with shape (n_points, 2)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # STEM Image
    ax_stem = axs[0]
    im_stem = ax_stem.imshow(stem_img, cmap='gray')
    ax_stem.scatter(X_scaled, Y_scaled, c='red', marker='o', label='Coordinates')
    # ax_stem.scatter(X * scaling_factor_x_stem, Y * scaling_factor_y_stem, c='red', marker='o', label='Coordinates')

    ax_stem.set_title('STEM Image IFFT', fontsize=20)
    divider_stem = make_axes_locatable(ax_stem)
    cax_stem = divider_stem.append_axes("right", size="5%", pad=0.05)
    cbar_stem = plt.colorbar(im_stem, cax=cax_stem)

    # HRTEM Image
    ax_hrtem = axs[1]
    im_hrtem = ax_hrtem.imshow(hrtem_img, cmap='gray')
    ax_hrtem.scatter(X_scaled, Y_scaled, c='red', marker='o', label='Coordinates')
    # ax_hrtem.scatter(X * scaling_factor_x_stem, Y * scaling_factor_y_stem, c='red', marker='o', label='Coordinates')

    ax_hrtem.set_title('STEM Image', fontsize=20)
    divider_hrtem = make_axes_locatable(ax_hrtem)
    cax_hrtem = divider_hrtem.append_axes("right", size="5%", pad=0.05)
    cbar_hrtem = plt.colorbar(im_hrtem, cax=cax_hrtem)

    plt.tight_layout()
    # plt.show()
    # plt.close()
    plt.close(fig)


def read_annotation_file(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    image_filename = root.find("filename").text
    image_path = root.find("filename").text
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    polygon_points = []
    names_and_colors = {}  # Store unique names and their associated colors

    for object_elem in root.findall(".//object"):
        name = object_elem.find("name").text
        polygon_elem = object_elem.find("polygon")
        polygon_elem = object_elem.find("polygon")

        if polygon_elem is not None:
            points = [(float(polygon_elem.find(f"x{i}").text), float(polygon_elem.find(f"y{i}").text))
                      for i in range(1, len(polygon_elem) // 2 + 1)]
            polygon_points.append(points)

            # Assign a color based on the name (you can customize the colors as needed)
            if name not in names_and_colors:
                names_and_colors[name] = 'C' + str(len(names_and_colors) % 10)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for points, name in zip(polygon_points, [obj.find("name").text for obj in root.findall(".//object")]):
        color = names_and_colors[name]
        polygon = Polygon(points, edgecolor=color, facecolor='none', label=name)
        ax.add_patch(polygon)

    # Add legend based on unique names and colors
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=names_and_colors[name], markersize=10, label=name)
        for name in names_and_colors]
    plt.legend(handles=handles, loc='upper right')

    plt.title(image_filename)
    # plt.savefig(annotation_file)
    # plt.show()
    # plt.close()
    plt.close(fig)


def save_pixel_intensity(saved_name_stem):
    im = Image.open(saved_name_stem, 'r')
    pix_val = list(im.getdata())
    pix_val_flat = [x for sets in pix_val for x in sets]
    mean_pixel_intensity = (np.mean(pix_val_flat))
    # print('Mean pixel intensity: ', mean_pixel_intensity)

    # print(np.max(pix_val_flat))
    # print(np.min(pix_val_flat))
    # params_dict.pop('STEM_Parameters')
    # print(params_dict)
    return np.mean(pix_val_flat)



def find_components_from_dir(xyz_path, surface_orientation = False):
    # Extract the filename from the path
    filename = os.path.basename(xyz_path)

    # Remove the file extension
    filename_no_ext = os.path.splitext(filename)[0]

    # Split the filename into parts using underscores
    filename_parts = filename_no_ext.split("_")

    if len(filename_parts) >= 2:
        c1, c2 = filename_parts[0], filename_parts[1]

        #######################################
        # c1, c2 = find_components_from_dir(xyz_path)
        component_list = ['LiF', 'Li2O', 'Li2CO3', 'LiOH']

        if surface_orientation == False:
            for i in component_list:
                if i in c1:
                    c1 = i
                if i in c2:
                    c2 = i

        #######################################

        return c1, c2
    else:
        # Handle the case where the filename doesn't have enough parts
        return None, None



# Extract points from the polygons
def extract_points_from_polygon(polygon):
    points = []
    for i in range(polygon.get_xy().shape[0]):
        x = polygon.get_xy()[i, 0]
        y = polygon.get_xy()[i, 1]
        points.append((x, y))
    return points





