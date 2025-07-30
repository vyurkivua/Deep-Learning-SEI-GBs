# Deep-Learning-SEI-GBs

This repository provides a deep learning framework for segmenting Solid-Electrolyte Interphase (SEI) grains and grain boundaries (GBs) in simulated HRTEM images using a hierarchical Transformer-based encoder with a Feature Pyramid Network (FPN) decoder. The approach addresses the challenges of low contrast, and fine-scale features by incorporating strong data augmentations, and custom evaluation metrics such as Matthews Correlation Coefficient (MCC) and Area Match (AM). The pipeline includes configurable training, modular code with PyTorch Lightning, and supports both training and inference on materials-science-inspired datasets.
To generate data, please check python package requirements from SEI-Simulation/gen_data/main.
Similrly to set up package requirements the DL model and train, use DL_model/main. To visualize the data run Visualize/visualize.py. Please make sure the sample annotated images are at correct folder.

1. Data Generation: To generate simulated SEI data:

cd SEI-Simulation/gen_data 

Install the required packages and run main.py to generate image-mask pairs.

2. Training the Deep Learning Model: To train or run inference:

cd DL_model

Install the necessary dependencies (e.g., from requirements.txt) and execute main.py.

4. Visualization:
To visualize model predictions and annotated masks:

cd Visualize
python visualize.py

Make sure your sample images are in the sample/images/ folder and corresponding masks are in sample/annotations/.

<img width="975" height="636" alt="image" src="https://github.com/user-attachments/assets/43a44b35-6ee8-4da2-bac1-a37c8f84abe2" />
