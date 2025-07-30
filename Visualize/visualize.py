import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from utils_visualize import Dataset, visualize_predictions


def main():
    # Set dataset paths
    dataset_path = '../sample'
    sample_images_path = os.path.join(dataset_path, 'images')
    sample_masks_path = os.path.join(dataset_path, 'annotations')

    output_folder = '../Annotation_Output'
    os.makedirs(output_folder, exist_ok=True)

    # 🔹 Create dataset and DataLoader
    paper_dataset_no_aug = Dataset(sample_images_path, sample_masks_path)
    paper_loader_no_aug = DataLoader(paper_dataset_no_aug, batch_size=1, shuffle=False, num_workers=0)

    classes = ['Background', 'Grain_Boundary', 'Li2CO3', 'LiOH', 'Li2O', 'LiF']
    n_classes = len(classes)

    # Generate class colors for visualization
    cmap = plt.cm.get_cmap('magma', n_classes)
    class_colors = {i: (np.array(cmap(i)[:3]) * 255).astype(np.uint8) for i in range(n_classes)}
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=classes[i],
                          markerfacecolor=class_colors[i] / 255, markersize=10) for i in range(n_classes)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(paper_loader_no_aug):
            images = images.to(device)
            images, masks = images.cpu(), masks.cpu()
            visualize_predictions(images, masks, filenames, class_colors, classes, patches, output_folder)


if __name__ == '__main__':
    main()
