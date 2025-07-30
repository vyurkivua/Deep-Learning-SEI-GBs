"""
SEI Segmentation Model Training Script
--------------------------------------

Author: Ishraque Zaman Borshon
Affiliation: The University of Arizona

Description:
This script trains a semantic segmentation model for SEI (Solid-Electrolyte Interphase)
region analysis using PyTorch Lightning. It utilizes packages including but not limited to pytorch lightning,
segmentation_models.pytorch, albumentations, matplotlib, skimage etc. and supports configurable architectures,
encoder, and augmentation pipeline via a JSON config file.

"""



import os
import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils import Dataset, get_training_augmentation, get_validation_augmentation, compute_class_weights, CLASSES
from sei_model import SEIModel


def main():
    # ------------------ Load config ------------------ #
    with open("config.json") as f:
        config = json.load(f)

    # ------------------ Paths ------------------ #
    dataset_path = config["dataset_path"]
    train_images = os.path.join(dataset_path, config["train_subdir"], config["image_dir"])
    train_masks = os.path.join(dataset_path, config["train_subdir"], config["mask_dir"])
    test_images = os.path.join(dataset_path, config["test_subdir"], config["image_dir"])
    test_masks = os.path.join(dataset_path, config["test_subdir"], config["mask_dir"])

    # ------------------ Dataset ------------------ #
    train_dataset = Dataset(train_images, train_masks, augmentation=get_training_augmentation())
    valid_dataset = Dataset(test_images, test_masks, augmentation=get_validation_augmentation())

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    # ------------------ Class weights ------------------ #
    class_weights = compute_class_weights(train_masks, len(CLASSES))

    # ------------------ Model ------------------ #
    model = SEIModel(
        arch=config["arch"],
        encoder_name=config["encoder_name"],
        in_channels=3,
        out_classes=len(CLASSES),
        classes=CLASSES,
        class_weights=class_weights,
    )

    # ------------------ Callbacks ------------------ #
    checkpoint = ModelCheckpoint(
        monitor='valid_dataset_iou',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='max',
        save_weights_only=True,
    )

    early_stop = EarlyStopping(
        monitor='valid_dataset_iou',
        patience=30,
        mode='max',
        verbose=True
    )

    # ------------------ Trainer ------------------ #
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint, early_stop],
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_loader, valid_loader)



if __name__ == "__main__":
    main()
