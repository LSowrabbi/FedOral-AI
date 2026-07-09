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

## Cross-Modality Analysis

To verify our model learns generalizable oral cancer features rather than 
modality-specific artifacts, we evaluated performance separately on 
clinical photographs and histopathology slides within our test set.

## Results (Centralized Baseline)
| Model | Parameters | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|----------|-----------|--------|----|---------|
| ResNet-18 | 11.3M | 88.6% | 92.7% | 88.6% | 90.6% | 95.8% |
| MobileNetV2 | 2.6M | 96.1% | 95.7% | 97.1% | 96.4% | 99.3% |
| **EfficientNet-B0 (progressive unfreeze)** | **4.3M** | **97.4%** | **97.8%** | **97.3%** | **97.5%** | **99.5%** |

Training: 50 epochs · Progressive unfreezing at epoch 10 · lr=0.001→0.0001
Dataset: 6,892 deduplicated images (ORCA + ashenafifasilkebede) · Seed=42

## Inference Benchmark (Apple Silicon MPS, 5 trials × 100 runs)
| Model | Latency (ms) | Throughput (img/s) |
|-------|--------------|---------------------|
| ResNet-18 | 4.11±0.02 | 243.5 |
| MobileNetV2 | 7.28±0.13 | 137.4 |
| EfficientNet-B0 | 9.44±0.10 | 105.9 |

Full analysis: `models/evaluate_by_modality.py` · Results: `results/modality_analysis/`

Training: 50 epochs · Progressive unfreezing at epoch 10 · lr=0.001→0.0001
Dataset: ORCA (1,700) + ashenafifasilkebede/Rahman et al. (5,192) · Deduplicated · Seed=42

Winner: EfficientNet-B0 → selected as backbone for federated learning system
Device: Apple Silicon MPS · Dataset: 1,700 images · Split: 70/15/15 · Seed: 42

## Status
- [x] ResNet-18, MobileNetV2, EfficientNet-B0 baselines
- [x] Progressive unfreezing — 97.4% accuracy (best model)
- [x] Grad-CAM explainability
- [x] Dataset deduplication — 6,892 clean images
- [x] Cross-modality validation (clinical photos vs histopathology)
- [x] Inference latency benchmarking
- [ ] Federated learning system
- [ ] Differential privacy

## Tech Stack
Python · PyTorch · Flower (flwr) · Docker · OpenCV
