import argparse
import numpy as np
import os
import time
import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from tools.plot import make_plot, get_roc_auc_score

import optuna

from dataset.dataset import XRaysDataset
from models.model import CustomModel
from run.trainer import fit

from torchvision import transforms

import logging
logger = logging.getLogger(__name__)

def get_transform():
   
    # 前処理の定義
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    return transform

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def check(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience  # patience 回以上改善しなければ True を返す

# 目的関数
def objective(args, trial):
    # ハイパーパラメータの探索
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    beta1 = trial.suggest_uniform("beta1", 0.85, 0.95)  # β1
    beta2 = trial.suggest_uniform("beta2", 0.995, 0.9999)  # β2
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)  # L2正則化
    eps = trial.suggest_loguniform("eps", 1e-10, 1e-6)

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    epochs = trial.suggest_int("epochs", 3, 50)
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    #momentum = trial.suggest_float("momentum", 0.5, 0.99) if optimizer_name == "SGD" else None

    # デバイスの決定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'device is {device}')

    # 前処理の読み込み
    transform = get_transform()

    # make the datasets
    XRayTrain_dataset = XRaysDataset(args, args.data_dir, args.class_numbers, None, transform, subset = 'train')
    train_percentage = 0.8
    #args.train_pc
    train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset, [int(len(XRayTrain_dataset)*train_percentage), len(XRayTrain_dataset)-int(len(XRayTrain_dataset)*train_percentage)])
    #test_dataset = XRaysDataset(args, args.data_dir, transform, subset = 'test')
    
    # make the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = not True)

    model = CustomModel(args.model_name, len(XRayTrain_dataset.all_classes)).to(device)

    # 実験に必要な関数の定義
    #criterion_name = 'BCEWithLogitsLoss'
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
        
    optimizer_name = 'Adam'
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        eps=eps
    )

    model.to(device)

    # Early Stopping の設定
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # 訓練ループ
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loader_examples_num = len(val_loader.dataset)
        probs = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
        gt    = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
        k=0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                #_, predicted = torch.max(outputs, 1)
                #correct += (predicted == labels).sum().item()
                #total += labels.size(0)

                probs[k: k + outputs.shape[0], :] = outputs.cpu()
                gt[   k: k + outputs.shape[0], :] = labels.cpu()
                k += outputs.shape[0]

        avg_val_loss = val_loss / len(val_loader)
        #accuracy = correct / total
        roc_auc = get_roc_auc_score(args, gt, probs)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")#, Val Acc: {accuracy:.4f}")

        # Early Stopping 判定
        if early_stopping.check(avg_val_loss):
            print("Early stopping triggered")
            break

    return roc_auc

def tuning_start(args):
    # Optuna の最適化実行
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(args, trial), n_trials=10)

    # 最適ハイパーパラメータの出力
    print("Best hyperparameters:", study.best_params)
