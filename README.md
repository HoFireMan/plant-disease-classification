# Plant Disease Classification Project (High-Accuracy Model)

---

### Description

This project leverages deep learning to build a high-accuracy classification model for plant leaf diseases, based on the public PlantVillage dataset. By employing an advanced transfer learning strategy—specifically, **Fine-tuning a ResNet50 model**—this solution achieves an outstanding accuracy of **99.47%** on the hold-out test set, surpassing previously published results on this dataset.

*   🚀 **See it in action on Kaggle:** [**Live Kaggle Notebook (99.47% Result)**](https://www.kaggle.com/code/hofireman/99-47-fine-tuned-resnet50-sota-result)
*   💻 **Explore the full source code on GitHub:** [**HoFireMan/plant-disease-classification**](https://github.com/HoFireMan/plant-disease-classification)
---

### Key Results

*   **Test Set Accuracy**: **99.47%**
*   **Model Architecture**: **ResNet50**
*   **Core Training Strategy**: **Fine-tuning**
*   **Model Interpretability**: Model decisions visualized using **Grad-CAM**

---

### Project Structure

The project is organized as follows:

```
.
│
├── 📁 evaluate/                      # Contains all evaluation-related scripts
│   ├── evaluate_All.py               # Unified, parameterized model evaluation script
│   └── grad_cam.py                   # Script for generating Grad-CAM heatmaps
│
├── 📁 Fine-tuning/                   # Contains training script for the final model
│   └── train_ResNet50_FineTune.py    # 🏆 Training script for the champion model
│
├── 📁 output/                        # Contains outputs from all experiments
│   └── 📁 train_ResNet50_FineTune_output/ # Dedicated output for the champion model
│       ├── resnet50_finetune_model_fold_3_best.pth  # Best model weights file
│       └── ...
│
├── 📁 PlantVillage/                  # Contains the original image dataset
│   └── ...
│
├── 📁 z_archived_scripts/            # Archived experimental scripts
│   └── ...
│
├── train_val_set.csv                 # Data definition for training and cross-validation
├── test_set.csv                      # Data definition for the hold-out test set
├── label_mapping.json                # Mapping from class names to indices
├── visualize_augmentations.py        # Utility script to generate data augmentation examples
├── get_baseline_avg_acc.py           # Utility script to re-validate K-Fold accuracy
├── README.md                         # Project documentation file
└── LICENSE                           # MIT License file
```

---

### Core Methodology

The high accuracy of this project is attributed to a rigorous and interconnected pipeline for data handling and model training.

#### **0. Download the Dataset**

This project requires the PlantVillage dataset. Please download it from the link below and place the `PlantVillage` folder into the root of this project directory before proceeding.

*   **Kaggle Dataset Link:** [**PlantVillage Dataset**](https://www.kaggle.com/datasets/emmarex/plantdisease)

#### **1. Data Cleaning and Preprocessing**
*   **Cleaning**: During the data loading phase, non-image system files (e.g., `.svn` files) were filtered out to ensure that only valid image formats like `.jpg`, `.jpeg`, and `.png` were fed into the training pipeline.
*   **Preprocessing**: All images were uniformly preprocessed before being input to the model: (1) Resized and center-cropped to `224x224` pixels; (2) Converted to PyTorch Tensors; (3) Normalized using ImageNet's mean and standard deviation.

#### **2. Training Approach Details**
Our champion model was trained using the following key techniques:
*   **Core Strategy - Fine-tuning**: Instead of training from scratch, we loaded a ResNet50 model pre-trained on ImageNet. After freezing all layers, we selectively **unfroze** the top two convolutional blocks (`layer3` and `layer4`) and replaced the final classifier head.
*   **Differential Learning Rates**: This was crucial for successful fine-tuning. We set a higher learning rate (`1e-4`) for the **newly replaced classifier head** to enable fast learning, while applying a very low learning rate (`1e-5`) to the **unfrozen convolutional layers** to refine their pre-existing knowledge without destroying it.
*   **Data Augmentation**: During training, we applied on-the-fly random augmentations to each image, including **Random Resized Crop**, **Random Horizontal Flip**, and **Color Jitter** (brightness, contrast, saturation), to significantly expand the dataset and improve the model's generalization ability.
*   **Robust Validation and Training Control**:
    *   **5-Fold Stratified Cross-Validation**: This method was used to obtain a more stable and reliable evaluation of the model's performance.
    *   **Early Stopping**: We continuously monitored the validation accuracy during training and automatically stopped the process if no improvement was seen for 5 consecutive epochs, preventing overfitting and selecting the best-performing model state.

---

### Installation

1.  **Clone the project**
    ```bash
    git clone https://github.com/HoFireMan/plant-disease-classification.git
    cd plant-disease-classification
    ```

2.  **Create and activate Conda environment**
    ```bash
    conda create --name plantdisease_env python=3.10 -y
    conda activate plantdisease_env
    ```

3.  **Install core dependencies**
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install pandas matplotlib seaborn opencv-python scikit-learn tqdm
    ```

---

### Usage

**Note:** All commands should be run from the project's root directory (`plant-disease-classification`).

#### 1. Training the Champion Model
```bash
python Fine-tuning/train_ResNet50_FineTune.py
```

#### 2. Evaluating the Champion Model
```bash
python evaluate/evaluate_All.py --model_name resnet50_finetune --fold 3
```

#### 3. Generating Grad-CAM Visualizations
```bash
python evaluate/grad_cam.py --fold 3 --num_images 5
```

---

### Acknowledgements, License, and Citations

#### **License**
*   **Code**: All code in this project is licensed under the Apache License 2.0**.
*   **Dataset**: The **PlantVillage dataset** is provided under the **CC0: Public Domain** license.

#### **Citations**
This project builds upon the following pioneering research:
1.  **PlantVillage Dataset**:
    > Hughes, D. P., & Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.
2.  **ResNet**:
    > He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*.
3.  **Grad-CAM**:
    > Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE international conference on computer vision*.
