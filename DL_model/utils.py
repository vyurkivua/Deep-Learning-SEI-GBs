import os
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
from skimage import measure

CLASSES = ['Background', 'Grain_Boundary', 'Li2CO3', 'LiOH', 'Li2O', 'LiF']
n_classes = len(CLASSES)

# Generate class colors for visualization
cmap = plt.cm.get_cmap('tab20', n_classes)
class_colors = {i: (np.array(cmap(i)[:3]) * 255).astype(np.uint8) for i in range(n_classes)}
patches = [plt.Line2D([0], [0], marker='o', color='w', label=CLASSES[i],
                      markerfacecolor=class_colors[i] / 255, markersize=10) for i in range(n_classes)]

class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        self.CLASSES = CLASSES
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('hrtem_image', 'label')) for image_id in self.ids]

        self.valid_ids = [
            image_id for image_id, mask_fp in zip(self.ids, self.masks_fps) if os.path.exists(mask_fp)
        ]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.valid_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('hrtem_image', 'label')) for image_id in self.valid_ids]

        self.class_map = {i: i for i in range(len(CLASSES))}
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask_remap = np.zeros_like(mask)

        for class_value, new_value in self.class_map.items():
            mask_remap[mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        image = image.transpose(2, 0, 1)
        return image, mask_remap

    def __len__(self):
        return len(self.valid_ids)

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf([
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ], p=0.9),
        A.OneOf([
            A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.9),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.9),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.PadIfNeeded(min_height=256, min_width=256)
    ])

def visualizes(image, mask):
    plt.figure(figsize=(12, 8))
    image = image.transpose(1, 2, 0)

    plt.subplot(121)
    plt.imshow(image)
    plt.title("Input Image")

    plt.subplot(122)
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(n_classes):
        color_mask[mask == i] = class_colors[i]
    plt.imshow(color_mask)
    plt.title("Ground Truth")

    plt.legend(handles=patches, loc='lower right', bbox_to_anchor=(0.4, -0.3),
               ncol=len(CLASSES), frameon=False)
    plt.tight_layout()
    plt.show()

def compute_class_weights(mask_dir, num_classes):
    pixel_counts = np.zeros(num_classes, dtype=int)
    for mask_fp in os.listdir(mask_dir):
        mask = cv2.imread(os.path.join(mask_dir, mask_fp), cv2.IMREAD_GRAYSCALE)
        for class_id in range(num_classes):
            pixel_counts[class_id] += np.sum(mask == class_id)

    total_pixels = np.sum(pixel_counts)
    weights = total_pixels / (num_classes * pixel_counts)
    return np.nan_to_num(weights, nan=0.0)

def mcc_score(tp, fp, tn, fn):
    n1 = tp * tn
    n2 = fp * fn
    n = n1 - n2

    d1 = tp + fp
    d2 = tp + fn
    d3 = tn + fp
    d4 = tn + fn
    d = np.sqrt(d1 * d2 * d3 * d4)

    mcc = n / (d + 1e-10)

    return mcc


def compute_am_score(gt_mask, pr_mask):
    """
    Computes the AM Score as:
      AM = (Number of correctly predicted foreground pixels) / (Total number of foreground pixels in GT)

    Only counts pixels where the ground truth is nonzero.
    A pixel is considered correctly predicted if:
         pr_mask == gt_mask   (for foreground pixels)
    """
    # Convert masks to NumPy arrays
    gt_mask_np = gt_mask.cpu().numpy()
    pr_mask_np = pr_mask.cpu().numpy()

    # Define foreground pixels (ignore background, assumed to be 0)
    gt_foreground = (gt_mask_np != 0)
    total_foreground = np.sum(gt_foreground)

    # Avoid division by zero: if no foreground, define AM score as 1.0 (or handle as needed)
    if total_foreground == 0:
        return 1.0

    # A pixel is correct if it is foreground and predicted exactly as ground truth
    correct_pixels = (gt_mask_np == pr_mask_np) & gt_foreground
    correct_count = np.sum(correct_pixels)

    am_score = correct_count / total_foreground
    return am_score


def overlay_mask(given_mask, class_colors, classes):
    # Convert mask to NumPy
    mask_np = given_mask.cpu().numpy()
    unique_classes = np.unique(mask_np)

    # Sort classes so class 1 is processed last
    sorted_classes = sorted(unique_classes, key=lambda x: (x == 1, x))

    # Create an RGBA overlay: shape (height, width, 4)
    full_mask = np.zeros((*mask_np.shape, 4), dtype=np.float32)

    # We'll draw class-1 contours last
    class_1_contours = []

    for class_value in sorted_classes:
        if class_value == 0:
            # Skip background class
            continue

        # Create a binary mask for the current class
        class_mask = (mask_np == class_value).astype(np.uint8)

        # Normalize color to [0,1]
        color = np.array(class_colors[class_value]) / 255.0

        # Find contours for this class
        contours = measure.find_contours(class_mask, level=0.5)

        # Save or draw contours
        if class_value == 1:
            class_1_contours.extend(contours)  # store class-1 contours
        else:
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0],
                         linewidth=2, color=color, linestyle='-')

        # Fill inside the class region with color
        # (i.e., only set the overlay in places where class_mask==1)
        full_mask[..., 0] += class_mask * color[0]  # Red channel
        full_mask[..., 1] += class_mask * color[1]  # Green channel
        full_mask[..., 2] += class_mask * color[2]  # Blue channel
        # Set alpha to 0.4 inside the contour
        full_mask[..., 3] += class_mask * 0.4

    # Finally, draw class-1 contours last
    for contour in class_1_contours:
        plt.plot(contour[:, 1], contour[:, 0],
                 linewidth=2, color=np.array(class_colors[1]) / 255.0,
                 linestyle='-')

    # Show the RGBA overlay, outside of contours alpha=0 => fully transparent
    plt.imshow(full_mask, interpolation='nearest')


def find_limits(atomic_structure):
    # Calculate the center of the image for zooming purposes
    center_x = atomic_structure.shape[1] // 2
    center_y = atomic_structure.shape[0] // 2
    zoom_in = 0.70
    # Calculate the width and height based on the zoom level
    zoom_width = atomic_structure.shape[1] * zoom_in
    zoom_height = atomic_structure.shape[0] * zoom_in

    # Calculate the new limits based on the center and zoom level
    xmin = center_x - zoom_width / 2
    xmax = center_x + zoom_width / 2
    ymin = center_y - zoom_height / 2
    ymax = center_y + zoom_height / 2

    return xmin, xmax, ymax, ymin


def visualize_predictions(images, gt_masks, pr_masks, filenames, class_colors, structure_dir, classes, output_folder):
    for idx, (image, gt_mask, pr_mask, filename) in enumerate(zip(images, gt_masks, pr_masks, filenames)):
        # Load atomic structure image
        atomic_structure = cv2.imread(os.path.join(structure_dir, filename.replace('hrtem_image', 'atomic_structure')))
        if atomic_structure is not None:
            atomic_structure = cv2.cvtColor(atomic_structure, cv2.COLOR_BGR2RGB)
        else:
            atomic_structure = np.zeros((256, 256, 3), dtype=np.uint8)

        plt.figure(figsize=(18, 10))

        # Subplot 1: Atomic Structure Image
        plt.subplot(241)
        plt.imshow(atomic_structure)
        plt.axis("off")
        xmin, xmax, ymax, ymin = find_limits(atomic_structure)
        plt.xlim(xmin, xmax)
        plt.ylim(ymax, ymin)

        # Subplot 2: Simulated Image
        plt.subplot(242)
        plt.imshow(image.cpu().numpy().transpose(1, 2, 0), cmap='gray')
        plt.axis("off")

        # Subplot 3: Ground Truth Mask
        plt.subplot(243)
        plt.imshow(image.cpu().numpy().transpose(1, 2, 0), cmap='gray')
        overlay_mask(gt_mask, class_colors, classes)
        plt.axis("off")

        # Subplot 4: Predicted Mask with AM score
        plt.subplot(244)
        plt.imshow(image.cpu().numpy().transpose(1, 2, 0), cmap='gray')
        overlay_mask(pr_mask, class_colors, classes)
        plt.axis("off")

        # Compute AM score and print on the figure
        am_score = compute_am_score(gt_mask, pr_mask)
        am_score_str = f"{am_score:.3f}"
        # Display the AM score on the top-right corner of subplot 4
        plt.text(0.95, 0.05, f"AM Score: {am_score_str}",
                 transform=plt.gca().transAxes, fontsize=14,
                 color='red', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # Save the composite output image with the AM score in the filename
        new_filename = filename.replace('hrtem_image_', '').replace('.png', f"_{am_score_str}.png")
        plt.subplots_adjust(bottom=0.02, top=0.98)
        plt.savefig(os.path.join(output_folder, new_filename), dpi=300, bbox_inches='tight')



        # Now, save individual subplots in Prediction_single_output folder:
        single_output_folder = 'Prediction_single_output'
        os.makedirs(single_output_folder, exist_ok=True)
        # Draw the canvas to update layout and renderer
        plt.gcf().canvas.draw()
        for i, ax in enumerate(plt.gcf().axes):
            # Get tight bounding box in figure coordinates and convert to inches
            bbox = ax.get_tightbbox(plt.gcf().canvas.get_renderer())
            bbox = bbox.transformed(plt.gcf().dpi_scale_trans.inverted())
            single_filename = new_filename.replace('.png', f"_{i}.png")
            plt.gcf().savefig(os.path.join(single_output_folder, single_filename), bbox_inches=bbox, dpi=300)

        plt.show()
        plt.close()