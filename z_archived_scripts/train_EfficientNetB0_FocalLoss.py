# Copyright (c) 2025 YI-AN YEH
# This project is licensed under the MIT License - see the LICENSE file for details.

# train_EfficientNetB0_FocalLoss.py

"""
Training script for the EfficientNet-B0 model using Focal Loss.
Completes the experimental matrix by testing the EfficientNet-B0 architecture with an advanced loss function.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 1. 全域設定參數 ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 15
N_SPLITS = 5
NUM_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-4

# --- 2. Focal Loss 類別定義 ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# --- 3. PyTorch 資料集類別 ---
class PlantDiseaseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        label = self.df.iloc[idx]['label_idx']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# --- 4. PyTorch 模型定義 ---
def get_model(num_classes=15, pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return model

# --- 5. 訓練與驗證函數 ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
    return running_loss / total_samples, (correct_predictions.double() / total_samples).item()

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
    return running_loss / total_samples, (correct_predictions.double() / total_samples).item()

# --- 6. 主執行函數 ---
def main(output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available(): print("Device name:", torch.cuda.get_device_name(0))

    data_transforms = {
        'train': transforms.Compose([...]), 'val': transforms.Compose([...])
    }
    data_transforms['train'] = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_transforms['val'] = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading data from train_val_set.csv...")
    train_val_df = pd.read_csv('train_val_set.csv')
    
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    X_kfold, y_kfold = train_val_df.index, train_val_df['label_idx']
    fold_val_accuracies = []

    print("Starting K-Fold Cross-Validation for EfficientNet-B0 with Focal Loss...")
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_kfold, y_kfold)):
        start_time = time.time()
        print("-" * 50)
        print(f"       TRAINING FOR FOLD {fold + 1} / {N_SPLITS}       ")
        print("-" * 50)

        train_df, val_df = train_val_df.iloc[train_ids], train_val_df.iloc[val_ids]
        train_dataset = PlantDiseaseDataset(train_df, transform=data_transforms['train'])
        val_dataset = PlantDiseaseDataset(val_df, transform=data_transforms['val'])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True)

        model = get_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
        
        # 使用 Focal Loss
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # 優化器只訓練分類層
        optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

        best_val_acc = 0.0
        epochs_no_improve = 0
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                print(f"Validation accuracy improved... Saving model...")
                best_val_acc = val_acc
                epochs_no_improve = 0
                model_save_path = os.path.join(output_dir, f'efficientnet_b0_focal_model_fold_{fold+1}_best.pth')
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
        
        fold_val_accuracies.append(best_val_acc)
        end_time = time.time()
        print(f"Fold {fold+1} finished in {(end_time - start_time)/60:.2f} minutes.")

    print("\n" + "="*50)
    print("       EfficientNet-B0 + Focal Loss CROSS-VALIDATION FINAL RESULTS       ")
    print("="*50)
    mean_accuracy = np.mean(fold_val_accuracies)
    std_accuracy = np.std(fold_val_accuracies)
    for i, acc in enumerate(fold_val_accuracies):
        print(f"Fold {i+1} Best Validation Accuracy: {acc:.4f}")
    print("-" * 50)
    print(f"EfficientNet-B0 + Focal Loss 5-折交叉驗證結果")
    print(f"平均驗證準確率: {mean_accuracy:.4f}")
    print(f"驗證準確率標準差: {std_accuracy:.4f}")
    print("-" * 50)

# --- 7. 程式進入點 ---
if __name__ == '__main__':
    output_dir = r'C:\Code\ClassHomeWork\plantdisease\output\train_EfficientNetB0_FocalLoss_output'
    output_filename = 'train_EfficientNetB0_FocalLoss_output.txt'
    output_filepath = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)

    original_stdout = sys.stdout 
    
    with open(output_filepath, 'w') as f:
        sys.stdout = f
        print(f"--- Script execution started at {time.ctime()} ---")
        try:
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError:
            pass
        main(output_dir)
        print(f"\n--- Script execution finished at {time.ctime()} ---")

    sys.stdout = original_stdout
    print(f"Script finished. Check the full log and models at: {output_dir}")