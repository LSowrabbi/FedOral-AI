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

## Results (Centralized Baseline)
| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Parameters |
|-------|----------|-----------|--------|----|---------|------------|
| ResNet-18 | 88.6% | 92.7% | 88.6% | 90.6% | 95.8% | 11.3M |
| EfficientNet-B0 | **93.7%** | **96.7%** | **93.0%** | **94.8%** | **98.9%** | 4.3M |

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
- [ ] Grad-CAM explainability visualizations
- [ ] Federated learning system (3 hospital nodes)
- [ ] Differential privacy (Opacus DP-SGD)
- [ ] arXiv preprint submission

## Tech Stack
Python · PyTorch · Flower (flwr) · Docker · OpenCV
