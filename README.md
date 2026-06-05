# An Application of U-Net Convolutional Network for 3D Brain Tumor Reconstruction and Segmentation using Reinforcement Learning

## 📌 Project Overview
This repository contains the source code and research components for advanced 3D Brain Tumor Semantic Segmentation and Volumetric Reconstruction utilizing the **BraTS 2020** dataset. 

The core architecture leverages a customized **3D U-Net Convolutional Neural Network** integrated with **Reinforcement Learning (RL)** policies to dynamically optimize loss weights, multi-scale feature selection, and regional refinement boundaries. This approach mitigates severe class imbalance inherent in medical imaging (Enhancing Tumor, Tumor Core, Whole Tumor) and enhances the spatial alignment of the 3D reconstructed output.

---

## 🛠️ System Architecture

The pipeline consists of three primary phases:
1. **Pre-processing (`Brats_Dataset/`):** NIfTI (.nii.gz) parsing, intensity normalization (Z-score mapping), adaptive cropping, and volume patching.
2. **Segmentation Backbone (`model/`):** Deep 3D U-Net featuring customized Encoder-Decoder blocks, Residual Squeeze-and-Excitation (`SqueezeExcitation.py`) links, and an optimized Hybrid Loss combining **Weighted Dice Loss** and Focal Cross-Entropy.
3. **RL Refinement Loop:** A Reinforcement Learning framework where an agent iteratively adjusts active contour thresholds, confidence masks, or loss-function balances based on local Dice Score and Hausdorff Distance feedback markers.

---

## 📁 Repository Structure

```text
├── Brats_Dataset/
│   ├── BratsSet.py                    # Custom PyTorch Dataset class for NIfTI processing
│   └── pre_processing.py              # Intensity scaling, slicing, and 3D data augmentation
├── model/
│   ├── model.py                       # Full 3D U-Net Network Assembly
│   ├── Encoder.py                     # Downsampling paths with deep feature extractors
│   ├── Decoder.py                     # Upsampling paths with skip-connection alignment
│   ├── Conv.py                        # Customized double/triple 3D Convolutional layers
│   ├── SqueezeExcitation.py           # Channel-wise attention mechanisms
│   ├── Weighted_Dice_Loss.py          # Class-balanced loss weights for target tumor sub-regions
├── checkpoints/                       # Model weight saves (*.pth)
├── outputs/                           # Segmentation mask inferences
├── results/                           # Metric logs, training history plots, and HTML representations
├── 3D_brain_tumor.ipynb               # End-to-end training and inference workbook
├── brats2020_3d_visualization.ipynb  # PyVista / Plotly interactive 3D mesh rendering
├── main.py                            # Execution script for training/evaluation pipelines
└── .gitignore                         # Project tracking exclusions (.zip, .pth, venv/)