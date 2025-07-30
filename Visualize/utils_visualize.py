import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import measure
from torch.utils.data import Dataset as BaseDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DL_model.utils import overlay_mask, find_limits

class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        self.CLASSES = classes
        self.ids = sorted(os.listdir(images_dir))  # Load image IDs in sorted order
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('hrtem_image', 'label')) for image_id in self.ids]

        # Filter out images for which the mask does not exist
        self.valid_ids = [
            image_id for image_id, mask_fp in zip(self.ids, self.masks_fps) if os.path.exists(mask_fp)
        ]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.valid_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('hrtem_image', 'label')) for image_id in self.valid_ids]

        self.background_class = 0  # Background is at index 0

        # Create a mapping for the classes
        self.class_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        # Create a blank mask to remap the class values
        mask_remap = np.zeros_like(mask)

        # Remap the mask according to the class map
        for class_value, new_value in self.class_map.items():
            mask_remap[mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        image = image.transpose(2, 0, 1)  # Convert HWC to CHW

        # Return the image, remapped mask, and filename
        return image, mask_remap, self.valid_ids[i]  # Add the filename here

    def __len__(self):
        return len(self.valid_ids)
    
def visualize_predictions(images, gt_masks, filenames, class_colors, classes, patches, output_folder):
    for idx, (image, gt_mask, filename) in enumerate(zip(images, gt_masks, filenames)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Display the image in both subplots
        for ax in axes:
            ax.imshow(image.cpu().numpy().transpose(1, 2, 0), cmap='gray')
            ax.axis("off")

        # Overlay mask in second subplot
        overlay_mask(gt_mask, class_colors, classes)

        # Create a custom legend
        legend = fig.legend(
            handles=patches[1:],  # skip background
            loc='lower center',
            ncol=len(class_colors) - 1,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.01)
        )

        # Adjust layout
        plt.subplots_adjust(wspace=0.05, bottom=0.05, top=0.95)

        # Save the figure
        new_filename = filename.replace('hrtem_image_', '').replace('.png', f"_gt.png")
        plt.savefig(os.path.join(output_folder, new_filename), dpi=300, bbox_inches='tight')

        print(f"Saved: {new_filename}")
        # plt.show()

        plt.close()
