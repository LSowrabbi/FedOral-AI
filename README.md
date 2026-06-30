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

## Status

- [x] Development environment setup (PyTorch 2.8, Flower, Opacus, MPS GPU)
- [x] Dataset loaded — 1,700 oral cancer images (CANCER / NON CANCER)
- [x] Data exploration notebook with visualizations
- [x] ResNet-18 model definition — forward pass working on Apple Silicon GPU
- [ ] Training loop with early stopping
- [ ] Grad-CAM explainability visualizations
- [ ] Federated learning system (3 hospital nodes)
- [ ] Differential privacy (Opacus DP-SGD)
- [ ] arXiv preprint submission

## Tech Stack
Python · PyTorch · Flower (flwr) · Docker · OpenCV
