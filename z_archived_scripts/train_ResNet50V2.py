# Copyright (c) 2025 YI-AN YEH
# This project is licensed under the MIT License - see the LICENSE file for details.

# train_ResNet50V2.py

"""
Plant Disease Classification Training Script using a pre-trained ResNet50 model.

This script implements transfer learning by loading a ResNet50 model pre-trained
on ImageNet, freezing its base layers, and replacing the final classifier head.

All print outputs are redirected and saved to an output file for logging purposes.
It safely uses multiprocessing for the DataLoader on Windows.
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
LEARNING_RATE = 2e-4 # 遷移學習通常使用較小的學習率1e-4

# --- 2. PyTorch 資料集類別 (與之前相同) ---
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

# --- 3. PyTorch 模型獲取函數 ---
def get_model(num_classes=15, pretrained=True):
    """
    載入預訓練的 ResNet50 模型並進行遷移學習設定。
    """
    # 從 torchvision.models 載入預訓練的 ResNet50
    # 注意: PyTorch 的 'resnet50' 是指 ResNet v1.5，這是最廣泛使用的標準版本
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
    
    # 凍結所有基礎層的參數，這樣在訓練時它們的權重不會被更新
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    # 獲取最後一層 (分類層) 的輸入特徵數
    num_ftrs = model.fc.in_features
    
    # 替換掉最後一層，換成我們自己的、適用於 15 個類別的分類層
    # 這個新的層的 requires_grad 預設為 True，所以只會訓練這一層的權重
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# --- 4. 訓練與驗證函數 (與之前相同) ---
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

# --- 5. 主執行函數 ---
def main():
    # 設定設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))

    # 定義資料增強
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # 載入資料
    print("Loading data from train_val_set.csv...")
    train_val_df = pd.read_csv('train_val_set.csv')
    
    # K-Fold 設定
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    X_kfold, y_kfold = train_val_df.index, train_val_df['label_idx']
    fold_val_accuracies = []

    print("Starting K-Fold Cross-Validation for ResNet50...")
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_kfold, y_kfold)):
        start_time = time.time()
        print("-" * 50)
        print(f"       TRAINING FOR FOLD {fold + 1} / {N_SPLITS}       ")
        print("-" * 50)

        train_df, val_df = train_val_df.iloc[train_ids], train_val_df.iloc[val_ids]
        train_dataset = PlantDiseaseDataset(train_df, transform=data_transforms['train'])
        val_dataset = PlantDiseaseDataset(val_df, transform=data_transforms['val'])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, persistent_workers=True)

        # 建立 ResNet50 模型
        model = get_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss()
        # 只將需要訓練的參數 (model.fc) 傳遞給優化器
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
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
                print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model...")
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'resnet50_model_fold_{fold+1}_best.pth')
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
        
        fold_val_accuracies.append(best_val_acc)
        end_time = time.time()
        print(f"Fold {fold+1} finished in {(end_time - start_time)/60:.2f} minutes.")

    print("\n" + "="*50)
    print("       ResNet50 CROSS-VALIDATION FINAL RESULTS       ")
    print("="*50)
    mean_accuracy = np.mean(fold_val_accuracies)
    std_accuracy = np.std(fold_val_accuracies)
    for i, acc in enumerate(fold_val_accuracies):
        print(f"Fold {i+1} Best Validation Accuracy: {acc:.4f}")
    print("-" * 50)
    print(f"ResNet50 5-折交叉驗證結果")
    print(f"平均驗證準確率: {mean_accuracy:.4f}")
    print(f"驗證準確率標準差: {std_accuracy:.4f}")
    print("-" * 50)

# --- 6. 程式進入點保護 與 輸出重定向 ---
if __name__ == '__main__':
    # 設定輸出目錄和檔案
    output_dir = r'C:\Code\ClassHomeWork\plantdisease\output'
    output_filename = 'train_ResNet50V2_output.txt'
    output_filepath = os.path.join(output_dir, output_filename)
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 儲存原始的 stdout
    original_stdout = sys.stdout 
    
    # 打開檔案並重定向 stdout
    with open(output_filepath, 'w') as f:
        sys.stdout = f
        
        print(f"--- Script execution started at {time.ctime()} ---")
        print(f"--- Output is being redirected to {output_filepath} ---")
        
        # 設定多程序啟動模式
        try:
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError:
            pass

        # 執行主函數
        main()
        
        print(f"\n--- Script execution finished at {time.ctime()} ---")

    # 恢復原始的 stdout
    sys.stdout = original_stdout
    print(f"Script finished. Check the full log at: {output_filepath}")