# FedOral-AI

Privacy-preserving federated learning for early oral cancer detection 
using distributed medical imaging.

## Overview
A research project exploring federated learning architectures that enable 
multiple clinical nodes to collaboratively train oral cancer detection 
models without sharing raw patient data — preserving privacy while 
improving diagnostic accuracy.

## Motivation
Oral cancer causes 180,000+ deaths annually. Non-tobacco causes — including 
chronic mechanical irritation from dental prosthetics — are underrepresented 
in current screening protocols. This project aims to build ML tools that 
detect lesions early, across underserved and non-traditional risk populations.

## Results (Centralized Baseline — EfficientNet-B0)
| Model | Dataset | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|---------|----------|-----------|--------|----|---------|
| ResNet-18 | 1,700 images | 88.6% | 92.7% | 88.6% | 90.6% | 95.8% |
| EfficientNet-B0 (frozen) | 6,892 images | 91.2% | 91.1% | 92.3% | 91.7% | 96.7% |
| **EfficientNet-B0 (progressive unfreeze)** | **6,892 images** | **97.4%** | **97.8%** | **97.3%** | **97.5%** | **99.5%** |

Training: 50 epochs · Progressive unfreezing at epoch 10 · lr=0.001→0.0001
Dataset: ORCA (1,700) + ashenafifasilkebede/Rahman et al. (5,192) · Deduplicated · Seed=42

Winner: EfficientNet-B0 → selected as backbone for federated learning system
Device: Apple Silicon MPS · Dataset: 1,700 images · Split: 70/15/15 · Seed: 42

## Status

- [x] Development environment (PyTorch 2.8, Flower, Opacus, MPS GPU)
- [x] Dataset loaded — 1,700 oral cancer images (CANCER / NON CANCER)
- [x] Data exploration notebook with visualizations
- [x] ResNet-18 model — forward pass on Apple Silicon GPU
- [x] Training loop with early stopping + weighted loss
- [x] Centralized baseline — 88.6% accuracy, AUC-ROC 0.958
- [x] EfficientNet-B0 comparison
- [x] Grad-CAM explainability visualizations
- [ ] Federated learning system (3 hospital nodes)
- [ ] Differential privacy (Opacus DP-SGD)


## Tech Stack
Python · PyTorch · Flower (flwr) · Docker · OpenCV
