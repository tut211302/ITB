import argparse
import os
import time
import sys
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from openpyxl import Workbook

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import hamming_loss
import matplotlib.pyplot as plt


from dataset.dataset import XRaysDataset
from models.model import CustomModel
from run.trainer import fit,test
from tools.evaluate import multilabel_evaluate, multilabel_accuracy

from torchvision import transforms

import logging
logger = logging.getLogger(__name__)

def q(text = ''): # easy way to exiting the script. useful while debugging
    print('> ', text)
    sys.exit()

def get_transform():
   
    # 前処理の定義
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    return transform

def ensemble_run_train(args):

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'device is {device}')

    class_indices = ['Effusion','Pneumothorax','Fibrosis','Consolidation','Atelectasis',
                    'Pneumonia','Infiltration','Mass','Nodule','Cardiomegaly',
                    'Emphysema','Pleural_Thickening','Edema','Hernia']
    class_indices = class_indices[:args.class_numbers]
    model_types = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    models = {}
    optimizers = {}

    for model_name, class_idx in zip(model_types, class_indices):

        # Load preprocessing
        transform = get_transform()

        # make the datasets
        XRayTrain_dataset = XRaysDataset(args, args.data_dir, 1 , class_idx, transform, subset = 'train')
        train_percentage = 0.8
        #args.train_pc
        train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset, [int(len(XRayTrain_dataset)*train_percentage), len(XRayTrain_dataset)-int(len(XRayTrain_dataset)*train_percentage)])
        #test_dataset = XRaysDataset(args, args.data_dir, args.class_numbers,transform, subset = 'test')
        
        logger.info('-----Initial Dataset Information-----')
        logger.info('num images in train_dataset   : {}'.format(len(train_dataset)))
        logger.info('num images in val_dataset     : {}'.format(len(val_dataset)))
        #logger.info('num images in test_dataset: {}'.format(len(test_dataset)))
        logger.info('-------------------------------------')

        # make the dataloaders
        batch_size = args.bs
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)
        #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = not True)
        
        logger.info('-----Initial Batchloaders Information -----')
        logger.info('num batches in train_loader: {}'.format(len(train_loader)))
        logger.info('num batches in val_loader  : {}'.format(len(val_loader)))
        #logger.info('num batches in test_loader : {}'.format(len(test_loader)))
        logger.info('-------------------------------------------')

        # モデルの作成
        model = CustomModel(model_name, 1).to(device)
        models[model_name] = model
        
        # オプティマイザの作成
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.5e-4, eps=1e-8)
        optimizers[model_name] = optimizer

        criterion = nn.BCEWithLogitsLoss().to(device)

        logger.info(f"\nTraining model: {model_name} for class {class_idx} (Binary Classification)")

        experiment_path = f'/home/fukuyama/ITB/experiment/{args.class_numbers}/{model_name}/'
        fit(args, device, train_loader, val_loader, model, criterion, optimizer, experiment_path)


    # make the datasets
    test_dataset = XRaysDataset(args, args.data_dir, 1, transform, subset = 'test')
    
    logger.info('-----Initial Dataset Information-----')
    logger.info('num images in test_dataset: {}'.format(len(test_dataset)))
    logger.info('-------------------------------------')

    # make the dataloaders
    batch_size = args.bs
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = not True)
    
    logger.info('-----Initial Batchloaders Information -----')
    logger.info('num batches in test_loader : {}'.format(len(test_loader)))
    logger.info('-------------------------------------------')

    # if args.ckpt == None:
    #     q('ERROR: Please select a checkpoint to load the testing model from')
        
    #args.experiment_path = f'/home/fukuyama/ITB/experiment/{args.class_numbers}/{args.model_name}/'
    
    # ===== テスト & 結果の統合 =====
    model_predictions = {name: [] for name in models.keys()}
    true_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            true_labels.append(labels.cpu().numpy())

            for model_name, class_idx in zip(models.keys(), class_indices):

                model = CustomModel(model_name, args.class_numbers).to(device)
                experiment_path = f'/home/fukuyama/ITB/experiment/{args.class_numbers}/{model_name}/'
                ckpt = torch.load(os.path.join(experiment_path, args.ckpt)) 
                model.load_state_dict(ckpt['model'])

                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()  # 確率値に変換
                model_predictions[model_name].append(preds)

    # 予測結果を統合
    true_labels = np.vstack(true_labels)
    for model_name in model_predictions:
        model_predictions[model_name] = np.vstack(model_predictions[model_name])

    threshold = args.threshold
    final_predictions = np.zeros_like(true_labels)

    for i, model_name in enumerate(models.keys()):
        final_predictions[:, i] = (model_predictions[model_name] > threshold).astype(int).reshape(-1)

    # 評価 (Hamming Loss, F1 Score)
    hamming = hamming_loss(true_labels, final_predictions)
    f1 = f1_score(true_labels, final_predictions, average="macro")

    logger.info(f"\nFinal Model - Hamming Loss: {hamming:.4f}, Macro F1 Score: {f1:.4f}")
    logger.info(f"Sample Predictions: {final_predictions[:5]}")  # 最初の5つを表示