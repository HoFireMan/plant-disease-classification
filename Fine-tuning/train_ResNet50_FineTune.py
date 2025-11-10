# Copyright (c) 2025 YI-AN YEH
# This project is licensed under the MIT License - see the LICENSE file for details.

# C:\Code\ClassHomeWork\plantdisease\Fine-tuning\train_ResNet50_FineTune.py

"""
Fine-tuning script for the ResNet50 model.

This script unfreezes the top layers (layer3 and layer4) of the pre-trained
ResNet50 model and trains them with a very low learning rate to adapt the
learned features to the plant disease dataset.
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 1. 路徑設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- 2. 全域設定參數 ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 15
N_SPLITS = 5
NUM_EPOCHS = 20 # Fine-tuning 通常收斂更快，不需要太多 epochs
PATIENCE = 5
# 為不同部分設定不同的學習率
BASE_LR = 1e-5     # 為解凍的卷積層設定極低的學習率
HEAD_LR = 1e-4     # 為新的分類頭設定較高的學習率

# --- 3. PyTorch 資料集類別 ---
class PlantDiseaseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # 使用 PROJECT_ROOT 來構建絕對路徑
        img_path = os.path.join(PROJECT_ROOT, self.df.iloc[idx]['filepath'])
        label = self.df.iloc[idx]['label_idx']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# --- 4. PyTorch 模型獲取函數 ---
def get_model_for_finetune(num_classes=15):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    
    # 1. 首先凍結所有層
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. 解凍頂部的卷積層 (layer3 和 layer4)
    print("Unfreezing layers: layer3, layer4")
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # 3. 替換分類頭
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# --- 5. 訓練與驗證函數 (與之前相同) ---
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
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    train_val_csv_path = os.path.join(PROJECT_ROOT, 'train_val_set.csv')
    print(f"Loading data from {train_val_csv_path}...")
    train_val_df = pd.read_csv(train_val_csv_path)
    
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    X_kfold, y_kfold = train_val_df.index, train_val_df['label_idx']
    fold_val_accuracies = []

    print("Starting K-Fold Cross-Validation for Fine-tuning ResNet50...")
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_kfold, y_kfold)):
        start_time = time.time()
        print("-" * 50)
        print(f"       FINE-TUNING FOR FOLD {fold + 1} / {N_SPLITS}       ")
        print("-" * 50)

        train_df, val_df = train_val_df.iloc[train_ids], train_val_df.iloc[val_ids]
        train_dataset = PlantDiseaseDataset(train_df, transform=data_transforms['train'])
        val_dataset = PlantDiseaseDataset(val_df, transform=data_transforms['val'])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True)

        model = get_model_for_finetune(num_classes=NUM_CLASSES).to(device)
        
        # 設定差異化學習率的優化器
        optimizer = optim.Adam([
            {'params': model.layer3.parameters(), 'lr': BASE_LR},
            {'params': model.layer4.parameters(), 'lr': BASE_LR},
            {'params': model.fc.parameters(), 'lr': HEAD_LR}
        ], lr=HEAD_LR) # 預設 lr 給 Adam
        
        criterion = nn.CrossEntropyLoss()
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
                model_save_path = os.path.join(output_dir, f'resnet50_finetune_model_fold_{fold+1}_best.pth')
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
    print("       Fine-tuned ResNet50 CROSS-VALIDATION FINAL RESULTS       ")
    print("="*50)
    mean_accuracy = np.mean(fold_val_accuracies)
    std_accuracy = np.std(fold_val_accuracies)
    for i, acc in enumerate(fold_val_accuracies):
        print(f"Fold {i+1} Best Validation Accuracy: {acc:.4f}")
    print("-" * 50)
    print(f"Fine-tuned ResNet50 5-折交叉驗證結果")
    print(f"平均驗證準確率: {mean_accuracy:.4f}")
    print(f"驗證準確率標準差: {std_accuracy:.4f}")
    print("-" * 50)

# --- 7. 程式進入點 ---
if __name__ == '__main__':
    output_dir = os.path.join(PROJECT_ROOT, 'output', 'train_ResNet50_FineTune_output')
    output_filename = 'train_ResNet50_FineTune_output.txt'
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