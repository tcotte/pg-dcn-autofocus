# Learning to Autofocus in Whole Slide Imaging via Physics-Guided Deep Cascade Networks

### *Unofficial PyTorch Implementation*

This repository provides an **unofficial implementation** of the paper:

> **Learning to Autofocus in Whole Slide Imaging via Physics-Guided Deep Cascade Networks**
> *Xia, Y., et al.*

The goal of this work is to reproduce and explore the paper’s proposed **physics-guided deep cascade network (PG-DCN)** for learning autofocus in **whole slide imaging (WSI)** systems.

---

## 🔍 Overview

Automatic and accurate autofocus is essential for high-quality whole-slide imaging in digital pathology. Traditional autofocus methods often rely on heuristic focus metrics that can be unreliable under varying imaging conditions.

This paper introduces a **deep learning–based autofocus method** that:

* Integrates **physical imaging priors** into the network design.
* Uses a **cascade architecture** that progressively refines focus estimation.
* Predicts **focus distance maps** or directly restores **in-focus images** from defocused inputs.

This repository aims to:

* Reproduce the PG-DCN model architecture.
* Provide training and inference scripts.
* Offer utilities for data preparation and evaluation.
* Serve as a reference implementation for researchers and practitioners in computational pathology, microscopy, and computational imaging.

---

## 🚀 Features

* **Physics-Guided Modules**
  Incorporates imaging physics (e.g., PSF modeling, defocus priors) as constraints or submodules.

* **Deep Cascade Architecture**
  Multiple refinement stages for progressively improving predictions.

* **Flexible Training**
  Configurable via YAML files; supports multi-GPU training.

* **Evaluation Tools**
  Includes sharpness metrics, reconstruction quality metrics (PSNR/SSIM), and visualization helpers.

---

## 📦 Installation

```bash
git clone https://github.com/tcotte/pg-dcn-autofocus.git
cd pg-dcn-autofocus

# Optionally create a conda environment
conda create -n pgdcn python=3.9
conda activate pgdcn

pip install -r requirements.txt
```

---

## 🔧 Usage

### **Training**

```bash
python scripts/train.py --config configs/config.yaml
```

### **Inference**

```bash
python scripts/infer.py --image_folder {dataset_path}
                        --classification_model_path {classification_model_path}
                        --negative_model_path {negative_model_path}
                        --positive_model_path {positive_model_path}
```

---

## 📚 Dataset

This implementation does **not** include proprietary WSI data.
You may prepare your own dataset following the format described in:

* The original paper
* The provided `data/README.md`

Datasets must include *paired or annotated defocus information*, depending on the training mode.

[//]: # (---)

[//]: # ()
[//]: # (## 📊 Results)

[//]: # ()
[//]: # (&#40;Replace with your achieved metrics, images, or qualitative comparisons once training is run.&#41;)

---

## 📝 Notes

* This repository is **unofficial**, meaning it is not associated with the original authors.
* Architectural choices and hyperparameters are based on the paper, but may differ in small implementation details.

---

## 🖊️ Citation

If you use this repository, please cite the **original paper**:

```
@article{xia2020learning,
  title={Learning to autofocus in whole slide imaging via physics-guided deep cascade networks},
  author={Xia, Y. and others},
  journal={...},
  year={2020}
}
```

---

## 🤝 Acknowledgments

* Thanks to the authors of the original paper for their contributions to computational imaging.
* Portions of this code are inspired by standard PyTorch best practices and open-source deep learning repositories.

