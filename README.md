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
🚧 Work in progress — building centralized baseline model (Phase 1)

## Planned Components
- [ ] Centralized CNN baseline (oral lesion classifier)
- [ ] Federated learning setup using Flower (flwr)
- [ ] Distributed node simulation via Docker
- [ ] Differential privacy layer
- [ ] Grad-CAM explainability visualizations

## Tech Stack
Python · PyTorch · Flower (flwr) · Docker · OpenCV
