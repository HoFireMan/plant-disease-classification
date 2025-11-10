# Copyright (c) 2025 YI-AN YEH
# This project is licensed under the MIT License - see the LICENSE file for details.

# C:\Code\ClassHomeWork\plantdisease\evaluate\evaluate_All.py

"""
Unified Evaluation Script for ALL Plant Disease Classification models.
This is the absolute final version, capable of evaluating all 8 experimental setups,
including the advanced fine-tuned models.
"""

import os

# --- 完整程式碼 ---
import sys
import json
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 15

class PlantDiseaseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path = os.path.join(PROJECT_ROOT, self.df.iloc[idx]['filepath'])
        label = self.df.iloc[idx]['label_idx']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class BaselineModel(nn.Module):
    def __init__(self, num_classes):
        super(BaselineModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(model_name, num_classes=15, pretrained=False):
    if 'baseline' in model_name:
        return BaselineModel(num_classes)
    elif 'resnet50' in model_name:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    elif 'effnet_b0' in model_name:
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main(args):
    model_config = {
        'baseline': {
            'dir': 'train_Baseline_output',
            'prefix': 'baseline_model_fold_'
        },
        'resnet50': {
            'dir': 'train_ResNet50V2_output',
            'prefix': 'resnet50_model_fold_'
        },
        'effnet_b0': {
            'dir': 'train_EfficientNetB0_output',
            'prefix': 'efficientnet_b0_model_fold_'
        },
        'baseline_focal': {
            'dir': 'train_Baseline_FocalLoss_output',
            'prefix': 'baseline_focal_model_fold_'
        },
        'resnet50_focal': {
            'dir': 'train_ResNet50V2_FocalLoss_output',
            'prefix': 'resnet50_focal_model_fold_'
        },
        'effnet_b0_focal': {
            'dir': 'train_EfficientNetB0_FocalLoss_output',
            'prefix': 'efficientnet_b0_focal_model_fold_'
        },
        'resnet50_finetune': {
            'dir': 'train_ResNet50_FineTune_output',
            'prefix': 'resnet50_finetune_model_fold_'
        },
        'resnet50_finetune_advanced': {
            'dir': 'train_ResNet50_FineTune_Advanced_output',
            'prefix': 'resnet50_finetune_advanced_model_fold_'
        }
    }

    config = model_config[args.model_name]
    
    model_dir = os.path.join(PROJECT_ROOT, 'output', config['dir'])
    BEST_MODEL_FILENAME = f"{config['prefix']}{args.fold}_best.pth"
    BEST_MODEL_PATH = os.path.join(model_dir, BEST_MODEL_FILENAME)
    
    JSON_MAPPING_PATH = os.path.join(PROJECT_ROOT, 'label_mapping.json')
    CONFUSION_MATRIX_PATH = os.path.join(model_dir, f'confusion_matrix_{args.model_name}_fold{args.fold}.png')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating Model: {args.model_name}, Fold: {args.fold} ---")
    print(f"Using device: {device}")

    test_csv_path = os.path.join(PROJECT_ROOT, 'test_set.csv')
    print(f"Loading hold-out test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    
    val_transforms = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = PlantDiseaseDataset(test_df, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Loading model architecture '{args.model_name}'...")
    print(f"Loading weights from: {BEST_MODEL_PATH}")
    model = get_model(args.model_name, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Predicting on Test Set"):
            inputs, labels = inputs.to(device), labels
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*50)
    print(f"       FINAL EVALUATION RESULTS ({args.model_name.upper()})       ")
    print("="*50)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Test Accuracy: {accuracy:.4f}\n")

    with open(JSON_MAPPING_PATH, 'r') as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}
    class_names = [label_mapping[i] for i in range(NUM_CLASSES)]

    print("Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    print("Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix for {args.model_name} on Test Set', fontsize=16)
    
    plt.savefig(CONFUSION_MATRIX_PATH, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate ALL trained plant disease models.")
    parser.add_argument('--model_name', type=str, required=True, 
                        choices=['baseline', 'resnet50', 'effnet_b0', 
                                 'baseline_focal', 'resnet50_focal', 'effnet_b0_focal',
                                 'resnet50_finetune', 'resnet50_finetune_advanced'], 
                        help='Name of the model configuration to evaluate.')
    parser.add_argument('--fold', type=int, required=True, 
                        help='The best fold number to evaluate (e.g., 1, 2, 3, 4, 5).')
    
    args = parser.parse_args()
    main(args)