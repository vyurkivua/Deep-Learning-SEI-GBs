"""
Model Definition Module
-----------------------

Credits:
- This model is integrated and customized from the `segmentation_models.pytorch` library by Pavel Yakubovskiy.
- segmentation_models.pytorch GitHub link: https://github.com/qubvel/segmentation_models.pytorch
"""

import os
import pandas as pd
import pytorch_lightning as pl
import segment_sei as sm
import torch
from torch.optim import lr_scheduler
from utils import mcc_score

class SEIModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, classes, class_weights, **kwargs):
        super().__init__()
        self.model = sm.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        print(arch, encoder_name, in_channels, out_classes, **kwargs)

        params = sm.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = sm.losses.DiceLoss(sm.losses.MULTICLASS_MODE, from_logits=True)

        self.train_metrics = []
        self.valid_metrics = []
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.class_accuracies = torch.zeros(self.number_of_classes, device=self.device)
        self.class_counts = torch.zeros(self.number_of_classes, device=self.device)
        self.classes = classes
        self.class_weights = class_weights

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch):
        image, mask = batch
        assert image.ndim == 4
        mask = mask.long()
        assert mask.ndim == 3

        logits_mask = self.forward(image)
        assert logits_mask.shape[1] == self.number_of_classes
        logits_mask = logits_mask.contiguous()

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)

        tp, fp, fn, tn = sm.metrics.get_stats(pred_mask, mask, mode="multiclass",
                                               num_classes=self.number_of_classes)

        # Calculate per-class loss
        per_class_loss = torch.zeros(self.number_of_classes, device=self.device)
        for class_idx in range(self.number_of_classes):
            per_class_loss[class_idx] = (mask == class_idx).float().mean() * loss.item()

        # Calculate per-class accuracy
        per_class_accuracy = tp / (tp + fp + fn + 1e-10)  # Avoid division by zero
        # per_class_mcc_score = mcc_score(tp, fp, tn, fn)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "per_class_loss": per_class_loss,
            # "per_class_mcc": per_class_mcc_score,  # Include per-class accuracy
            "per_class_accuracy": per_class_accuracy,  # Include per-class accuracy
        }

    def shared_epoch_end(self, outputs, stage):
        # Calculate average loss for the epoch
        avg_loss = torch.mean(torch.stack([x["loss"] for x in outputs]))
        # print('outputs', outputs)

        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Calculate per-class loss
        total_per_class_loss = torch.zeros(self.number_of_classes, device=self.device)
        for output in outputs:
            total_per_class_loss += output.get("per_class_loss",
                                               torch.zeros(self.number_of_classes, device=self.device))

        per_class_average_loss = total_per_class_loss / len(outputs)

        # Calculate IoU and accuracy
        per_image_iou = sm.metrics.iou_score(tp, fp, fn, tn, class_weights=self.class_weights, reduction="weighted")
        dataset_iou = sm.metrics.iou_score(tp, fp, fn, tn, class_weights=self.class_weights, reduction="weighted")
        accuracy = sm.metrics.accuracy(tp, fp, fn, tn, class_weights=self.class_weights, reduction="weighted")
        per_class_accuracy = mcc_score(tp, fp, tn, fn)

        # Calculate mean per-class accuracy
        mean_per_class_accuracy = per_class_accuracy.mean(dim=0)

        # Store mean per-class accuracies in self.class_accuracies
        if not hasattr(self, 'class_accuracies'):
            self.class_accuracies = torch.zeros(self.number_of_classes, device=self.device)

        # avg_per_class_accuracy = mean_per_class_mcc_score.detach()  # Save mean accuracies
        avg_per_class_accuracy = mean_per_class_accuracy.detach()  # Save mean accuracies
        # print(accuracy, per_class_accuracy)

        metrics = {
            f"{stage}_per_image_iou": per_image_iou.item(),
            f"{stage}_dataset_iou": dataset_iou.item(),
            f"{stage}_dataset_accuracy": accuracy.item(),
        }

        custom_metrics = {
            f"{stage}_dataset_iou": dataset_iou.item(),
            f"{stage}_loss": avg_loss.item(),
            f"{stage}_dataset_accuracy": accuracy.item(),
            **{f"{stage}_class_{i}_loss": per_class_average_loss[i].item() for i in range(self.number_of_classes)},
            **{f"{stage}_class_{i}_mcc": avg_per_class_accuracy[i].item() for i in range(self.number_of_classes)},
            # Average per-class accuracy
            # **{f"{stage}_class_{i}_accuracy": avg_per_class_accuracy[i].item() for i in range(self.number_of_classes)},  # Average per-class accuracy
        }

        if stage == "train":
            self.train_metrics.append(custom_metrics)
        else:
            self.valid_metrics.append(custom_metrics)

        self.log_dict(metrics, prog_bar=True)

        # Reset class accumulators after epoch
        if stage == "train":
            self.class_accuracies = torch.zeros(self.number_of_classes, device=self.device)
            self.class_counts = torch.zeros(self.number_of_classes, device=self.device)

    def on_fit_end(self):
        # Create a list to hold all the data
        history_data = []

        # Loop through epochs to collect metrics
        for epoch in range(len(self.train_metrics)):
            train_metrics = self.train_metrics[epoch]
            valid_metrics = self.valid_metrics[epoch]

            # Collect general metrics
            epoch_data = {
                "Epoch": epoch + 1,
                "Train Accuracy": train_metrics.get("train_dataset_accuracy", None),
                "Validation Accuracy": valid_metrics.get("valid_dataset_accuracy", None),
                "Train Loss": train_metrics.get("train_loss", None),
                "Validation Loss": valid_metrics.get("valid_loss", None),
            }

            # Collect per-class metrics
            for i in range(self.number_of_classes):
                epoch_data[f"Train Accuracy {self.classes[i]}"] = train_metrics.get(f"train_class_{i}_mcc", None)
                epoch_data[f"Validation Accuracy {self.classes[i]}"] = valid_metrics.get(f"valid_class_{i}_mcc",
                                                                                         None)
                epoch_data[f"Train Loss {self.classes[i]}"] = train_metrics.get(f"train_class_{i}_loss", None)
                epoch_data[f"Validation Loss {self.classes[i]}"] = valid_metrics.get(f"valid_class_{i}_loss", None)

            # Append epoch data to the history list
            history_data.append(epoch_data)

        # Convert to DataFrame
        history_df = pd.DataFrame(history_data)

        # Save to CSV
        self.output_csv_path = os.path.join(self.trainer.log_dir, 'training_history.csv')
        history_df.to_csv(self.output_csv_path, index=False)

        print(f"Training history saved to {self.output_csv_path}")

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch)
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch)
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


