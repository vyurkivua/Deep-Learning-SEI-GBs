# Deep-Learning-SEI-GBs

This repository provides a deep learning framework for segmenting Solid-Electrolyte Interphase (SEI) grains and grain boundaries (GBs) in simulated HRTEM images using a hierarchical Transformer-based encoder with a Feature Pyramid Network (FPN) decoder. The approach addresses the challenges of low contrast, and fine-scale features by incorporating strong data augmentations, and custom evaluation metrics such as Matthews Correlation Coefficient (MCC) and Area Match (AM). The pipeline includes configurable training, modular code with PyTorch Lightning, and supports both training and inference on materials-science-inspired datasets.

<img width="975" height="636" alt="image" src="https://github.com/user-attachments/assets/43a44b35-6ee8-4da2-bac1-a37c8f84abe2" />
