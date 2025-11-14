---

Unofficial Implementation of *Whole Slide Imaging System Using Deep Learning-Based Automated Focusing*

**Repository Name**: *wsi_autofocus*
**Authors**: Tristan Cotte
**License**: MIT 
**Based on Paper**:

> Dastidar T. R., Ethirajan R. (2019) *Whole Slide Imaging System Using Deep Learning-Based Automated Focusing.* Biomedical Optics Express 11(1): 480-491. ([PMC][1])

---

## 🧪 Overview

This repository contains an **unofficial** implementation of the autofocus component described in the paper, along with supporting dataset loaders, training scripts, and an example pipeline for integrating with whole slide image (WSI) acquisition.

The original system used a CNN (based on MobileNetV2) to estimate the defocus distance from a difference-image (two frames offset in Z) and thereby assist a low-cost whole-slide imaging scanner. ([PMC][1])

This implementation aims to provide a flexible PyTorch-based framework for research and development (not clinical use).

---

## 📦 Repository Structure

```
wsi_autofocus/
│   README.md
│   requirements.txt
│               
├───datasets/
│       base_dataset.py
│       preprocessing.py
│       transforms.py
│       
├───scripts/
│       model.py
│       predict.py
│       train.py
│       
└───utils/
        logger.py
        loss.py
        system.py
```

---

## 🔍 Key Components

### 1. Dataset & Preprocessing

1. Acquisition of Z-Stack Images

For each XY position on the slide, the paper captures a small sequence of images along the Z-axis:
{I(z−kΔz),…,I(z),…,I(z+kΔz)}
{I(z−kΔz),…,I(z),…,I(z+kΔz)}

Where:
- I(z) is an image acquired at focal position z
- Δz is a fixed step size (e.g., 1–3 µm)
- k determines how many planes are included around the central position

This short Z-sweep ensures that the model observes how image structures change as the lens moves closer or further from focus.
2. Construction of Difference Images

The original paper highlights that the difference between adjacent Z-planes contains the most discriminative focus information.

For each position, a difference image is computed:
D(z)=I(z+Δz)−I(z)
D(z)=I(z+Δz)−I(z)

Near the correct focus, images change smoothly → difference is small.
Far from focus, blur changes rapidly → difference is large.


3. Ground-Truth Label Assignment

Each difference image is paired with a focus offset label, defined as:
y=zoptimal−z

Where:
- zoptimalis the true focus location (determined experimentally)
- z is the plane from which the difference image was derived

Thus, the network learns a regression function that maps difference patterns → distance from optimal focus.
4. Intensity Normalization

Microscopy images may vary significantly in brightness and contrast due to:
- Tissue thickness 
- Staining intensity 
- Illumination non-uniformity

To reduce these variations, images are normalized using a channel-wise transformation:
Inorm=(I−μ)/σ

This ensures that focus prediction depends on structural differences rather than intensity fluctuations.


### 2. Model Architecture

* Based on MobileNetV2 (or a variant) with final dense layer for regression of defocus distance. ([PMC][1])
* Loss function: smooth L1 loss (Huber) weighted by patch standard deviation and defocus distance. ([PMC][1])
* Two-phase training:

  * Phase 1: Only patch standard deviation weighting.
  * Phase 2: Also include distance-based weighting. ([PMC][1])

### 3. Inference / Slide Scanning Loop

* Uses previous field-of-view focal position as pivot.
* For each new tile: move stage to a starting position below pivot, capture two images at known Δ, compute difference, predict defocus, move stage accordingly (compensate for backlash etc.). ([PMC][1])
* Optionally: repeat if prediction uncertainty is above threshold.

---

## 🎯 Quick Start

1. Clone the repo:

   ```bash
   git clone https://github.com/tcotte/wsi_autofocus.git
   cd wsi_autofocus
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your data:

* Loads Z-stack image fields of view with known defocus distances as shown below: 

```
root_dataset/
│   
├───X/
│   ├───train/
│   │   ├───position_xy_0/
│   │   │       postion_z_0.jpg
│   │   │       postion_z_1.jpg
│   │   │       postion_z_2.jpg
│   │   │       ...
│   │   │       
│   │   ├───position_xy_1/
│   │   │       postion_z_0.jpg
│   │   │       postion_z_1.jpg
│   │   │       postion_z_2.jpg
│   │   │       ...
│   │
│   ├───test/
│   │   ├───position_xy_0/
│   │   │       postion_z_0.jpg
│   │   │       postion_z_1.jpg
│   │   │       postion_z_2.jpg
│   │   │       ...
│   │   │       
│   │   ├───position_xy_1/
│   │   │       postion_z_0.jpg
│   │   │       postion_z_1.jpg
│   │   │       postion_z_2.jpg
│   │   │       ...
│               
└───y/
        train.xlsx                           # to be generated with datasets/preprocessing.py
        test.xlsx                     # to be generated with datasets/preprocessing.py
```

Defocus positions (specified in µm) are encoded in `make` Exif metadata.

* Computes the difference image: given two frames at (z_1) and (z_2 = z_1 + \Delta), difference (I = I_2 - I_1). ([PMC][1])
This computation can be done using the command below:
   ```bash
   python scripts/preprocessing.py \
    --train_dataset_folder "<root_dataset>/X/train" \
    --test_dataset_folder "<root_dataset>/X/test" \
    --output_folder "<root_dataset>/y"
   ```
4. Train the model:

   ```bash
   python scripts/train.py \
    --train_set "<root_dataset>/X/train" \
    --test_set "<root_dataset>/X/test" \
    --run_name "<run_name>" \
    --epochs 100 \
    --batch-size 128 
   ```
5. Evaluate:

   ```bash
   python scripts/predict.py \
    --image_folder "<your_image_test_dataset>" \
    --model_path "<your_model_path>" \
    --excel_path "<labelfile_test_dataset>"
   ```
---

## ✅ Results & Validation

In the original work, the authors reported a mean absolute focusing error of ~0.19 µm (same protocol) and ~0.25 µm (different staining protocol) using their two-phase method. ([PMC][1])
Your results may vary depending on data, objective, and hardware.

---

## ⚠️ Limitations & Disclaimers

* **Unofficial**: This is *not* the official implementation and has not been validated for clinical diagnostic use.
* Hardware specifics (stage backlash, objective depth-of-field, motion accuracy) significantly affect performance.
* The original dataset and microscope stage details differ from general use; adaptation may be required.
* Use at your own risk; ensure compliance with regulations if applied in real-world systems.

---

## 🧩 Future Work / Extensions

* Support for other CNN backbones (e.g., MobileNetV3, EfficientNet)
* Incorporation of deep learning single-shot autofocus (no two-frame difference)
* Real-time GPU deployment on low-cost edge hardware
* Integration with whole-slide image stitching pipelines
* Adaptation to different magnifications (20×, 40×), objective types, and staining protocols

---

## 📄 References

* Dastidar T. R., Ethirajan R. (2019) *Whole Slide Imaging System Using Deep Learning-Based Automated Focusing.* Biomed Opt Express 11(1):480–491. ([PMC][1])
* MobileNetV3 architecture: MobileNetV3 – A.Howard et al., 2019.
* Dataset description and autofocus background in original paper.

---

## Acknowledgements

* Thanks to the original authors for open access publication.
* Thanks to the open-source community for inspiration and tools.
* This project is inspired by the intersection of digital pathology and deep learning, aiming to lower the barrier for whole slide imaging in resource-constrained settings.

---

If you have any questions, issues or suggestions — feel free to open an issue or submit a pull request.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6968754/?utm_source=chatgpt.com "Whole slide imaging system using deep learning-based automated focusing - PMC"
