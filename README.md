# Dual-Stream-Medical-Image-Detection
A novel dual-stream CNN for detecting synthetic medical images combining spatial and frequency features. Achieves 92.88% accuracy on Brain MRI dataset (6,528 images). Q1 journal paper in preparation. Complete experimental code and results included.
# A Novel Dual-Stream Convolutional Network for Synthetic Medical Image Detection in Radiology

[![Paper Status](https://img.shields.io/badge/Paper-In%20Preparation-yellow)](https://github.com/dralamori/Dual-Stream-Medical-Image-Detection)
[![Dataset](https://img.shields.io/badge/Dataset-Brain%20MRI-blue)](https://github.com/dralamori/Dual-Stream-Medical-Image-Detection)
[![Test Accuracy](https://img.shields.io/badge/Accuracy-92.88%25-brightgreen)](https://github.com/dralamori/Dual-Stream-Medical-Image-Detection)

## 📊 Overview

This repository contains the complete experimental code and results for our Q1 journal paper on synthetic medical image detection. Our proposed dual-stream convolutional neural network combines spatial semantic features with frequency-domain artifacts to achieve state-of-the-art detection performance.

**Key Achievement:** 92.88% test accuracy on 6,528 Brain MRI images (3,264 real + 3,264 synthetic)

---

## 🎯 Main Results

| Model | Test Accuracy | Improvement over Baseline |
|-------|---------------|---------------------------|
| **Dual-Stream (Ours)** | **92.88%** | **Baseline** |
| ResNet18 (Standard) | 86.91% | **+5.97%** |
| Stream 1 Only (Spatial) | 65.70% | **+27.18%** |
| Stream 2 Only (Frequency) | 75.27% | **+17.61%** |

---

## 🏗️ Architecture

Our dual-stream architecture consists of:

1. **Stream 1 (Spatial):** ResNet18 pretrained on ImageNet → 512-dimensional features
   - Captures anatomical structures, tissue textures, and semantic patterns
   
2. **Stream 2 (Frequency):** Custom CNN → 128-dimensional features
   - Identifies frequency-domain artifacts from synthetic generation

3. **Fusion Layer:** Concatenation (640 features) → Dense layers → Binary classification

---

## 📁 Repository Contents
```
├── Medical_Image_Detection.ipynb    # Complete experimental code (26 cells)
├── figures/                          # Publication-quality figures
│   ├── architecture_diagram.png     # Figure 1: Network architecture
│   ├── training_curves.png          # Figure 2: Training history
│   ├── comparison_chart.png         # Figure 3: Model comparison
│   ├── results_table.png            # Table 1: Quantitative results
│   ├── sample_images.png            # Figure 4: Dataset samples
│   └── detailed_confusion_matrix.png # Figure 5: Confusion matrix
├── models/
│   └── dual_stream_cnn_full.pth     # Trained model weights
└── docs/
    └── PROJECT_DOCUMENTATION.txt     # Complete documentation
```

---

## 🔬 Experimental Setup

**Dataset:**
- Source: Brain Tumor MRI Classification Dataset
- Total Images: 6,528 (3,264 Real + 3,264 Synthetic)
- Train/Test Split: 80/20 (5,222 / 1,306)
- Image Size: 224×224×3
- Synthetic Generation: Latent Diffusion Model (LDM)

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Batch Size: 32
- Epochs: 10
- Hardware: 2× NVIDIA RTX 3090 (24GB)
- Framework: PyTorch 1.7.0

---

## 📈 Performance Metrics

**Our Dual-Stream Model:**
- **Test Accuracy:** 92.88%
- **Precision (Real):** 0.940
- **Recall (Real):** 0.916
- **Precision (Synthetic):** 0.918
- **Recall (Synthetic):** 0.942
- **F1-Score:** 0.929

---

## 🚀 Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dralamori/Dual-Stream-Medical-Image-Detection/blob/main/Medical_Image_Detection.ipynb)

### 2. Run All Cells

The notebook contains 26 cells covering:
- Cell 1-4: Dataset preparation
- Cell 5-6: Synthetic image generation (LDM)
- Cell 7-9: Dual-stream model architecture
- Cell 10-13: Model training
- Cell 14-19: Baseline comparisons and ablation studies
- Cell 20-26: Visualization and documentation

---

## 📄 Paper Information

**Title:** A Novel Dual-Stream Convolutional Network for Synthetic Medical Image Detection in Radiology

**Authors:** Rafid Ahmed Khalid Alamori et al.

**Status:** In preparation for Q1 journal submission

**Target Journals:**
- IEEE Transactions on Information Forensics and Security
- IEEE Transactions on Medical Imaging  
- Medical Image Analysis (Elsevier)
- Pattern Recognition (Elsevier)

**Paper Contributions:**
1. Novel dual-stream architecture combining spatial and frequency features
2. Comprehensive ablation study proving necessity of both streams
3. Large-scale evaluation on 6,528 medical images
4. Significant improvement (+5.97%) over ResNet18 baseline
5. Addresses critical problem of AI-generated medical image fraud

---

## 📊 Key Findings

### Ablation Study Results
Our ablation study demonstrates that **both streams are essential**:
- Removing spatial stream: **-27.18%** accuracy drop
- Removing frequency stream: **-17.61%** accuracy drop

This proves that spatial and frequency features provide complementary discriminative signals.

### Baseline Comparison
Our dual-stream architecture **outperforms** standard transfer learning:
- ResNet18: 86.91% → **Ours: 92.88%** (+5.97%)
- ResNet101: 95.39% (trained on different dataset)
- VGG16: 93.65% (trained on different dataset)

---

## 🔍 Reproducibility

All experimental results are **fully reproducible**:
- ✅ Complete code provided (26 cells)
- ✅ Dataset publicly available
- ✅ Trained model weights included
- ✅ Random seeds fixed
- ✅ Hyperparameters documented

---

## 📧 Contact

**Dr. Rafid Ahmed Khalid Alamori**
- 🏛️ Assistant Professor, University of Mosul, Iraq
- 🏛️ Head of Cyber Security Engineering Techniques Department
- 🏢 Founder, Golden Net AI

For questions about this research, please open an issue or contact via GitHub.

---

## 📖 Citation

**Paper in preparation.** Citation will be updated upon publication.
```bibtex
@article{alamori2025dualstream,
  title={A Novel Dual-Stream Convolutional Network for Synthetic Medical Image Detection in Radiology},
  author={Alamori, Rafid Ahmed Khalid Alamori},
  journal={[Journal Name]},
  year={2025},
  note={In preparation}
}
```

---

## 🙏 Acknowledgments

This research was conducted at:
- Department of Cyber Security Engineering
- Al-Qabas Private College, Iraq
- Golden Net AI Consulting Company

---

## 📜 License

This code is released for academic research purposes. Please cite our paper if you use this code.

---

**Last Updated:** November 11, 2025  
**Repository Status:** Complete experimental results | Paper writing in progress  
**Next Steps:** Submit to Q1 journal after paper completion

---

⭐ **Star this repository if you find it useful!**
