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

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
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

def run_train(args):

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'device is {device}')

    # Load preprocessing
    transform = get_transform()

    # make the datasets
    XRayTrain_dataset = XRaysDataset(args, args.data_dir, args.class_numbers, transform, subset = 'train')
    train_percentage = 0.8
    #args.train_pc
    train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset, [int(len(XRayTrain_dataset)*train_percentage), len(XRayTrain_dataset)-int(len(XRayTrain_dataset)*train_percentage)])
    test_dataset = XRaysDataset(args, args.data_dir, args.class_numbers,transform, subset = 'test')
    
    logger.info('-----Initial Dataset Information-----')
    logger.info('num images in train_dataset   : {}'.format(len(train_dataset)))
    logger.info('num images in val_dataset     : {}'.format(len(val_dataset)))
    logger.info('num images in test_dataset: {}'.format(len(test_dataset)))
    logger.info('-------------------------------------')

    # make the dataloaders
    batch_size = args.bs
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = not True)
    
    logger.info('-----Initial Batchloaders Information -----')
    logger.info('num batches in train_loader: {}'.format(len(train_loader)))
    logger.info('num batches in val_loader  : {}'.format(len(val_loader)))
    logger.info('num batches in test_loader : {}'.format(len(test_loader)))
    logger.info('-------------------------------------------')

    # sanity check
    # if len(XRayTrain_dataset.all_classes) != 15: # 15 is the unique number of diseases in this dataset
    #     q('\nnumber of classes not equal to 15 !')

    a,b = train_dataset[0]
    logger.info('we are working with \nImages shape: {} and \nTarget shape: {}'.format( a.shape, b.shape))

    model = CustomModel(args.model_name, len(XRayTrain_dataset.all_classes)).to(device)

    # define the loss function
    if args.loss_func == 'BCE':
        criterion_name = 'BCEWithLogitsLoss'
        criterion = nn.BCEWithLogitsLoss().to(device)
        
    optimizer_name = 'Adam'
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.5e-4,
        eps=1e-8
    )
    logger.info('-----Hyperparameter Information -----')
    logger.info(f"criterion is {criterion_name}.")
    logger.info(f"optimizer is {optimizer_name}.")
    logger.info(f"learning rate is {args.lr}.")

    # making empty lists to collect all the losses
    #losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}
    
    fit(args, device, train_loader, val_loader, model, criterion, optimizer)
    
def run_test(args):

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'device is {device}')

    # Load preprocessing
    transform = get_transform()

    # make the datasets
    test_dataset = XRaysDataset(args, args.data_dir, args.class_numbers, transform, subset = 'test')
    
    logger.info('-----Initial Dataset Information-----')
    logger.info('num images in test_dataset: {}'.format(len(test_dataset)))
    logger.info('-------------------------------------')

    # make the dataloaders
    batch_size = args.bs
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = not True)
    
    logger.info('-----Initial Batchloaders Information -----')
    logger.info('num batches in test_loader : {}'.format(len(test_loader)))
    logger.info('-------------------------------------------')

    # sanity check
    model = CustomModel(args.model_name, args.class_numbers).to(device)

    # if args.ckpt == None:
    #     q('ERROR: Please select a checkpoint to load the testing model from')
        
    logger.info('checkpoint loaded: {}'.format(args.ckpt))
    ckpt = torch.load(os.path.join(args.experiment_path, args.ckpt)) 
    model.load_state_dict(ckpt['model'])

    preds, labels = test(args, model, test_loader, device)

    # evaluate
    if args.class_numbers == 1:
        
        y_true = labels
        y_pred = preds
        y_pred = np.array(preds)
        y_pred = np.where(y_pred > args.threshold, 1.0, 0.0).tolist()
        cm = confusion_matrix(y_true, y_pred)
        TP, FP, TN, FN = cm.ravel()

        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        
        print(f"混同行列:\n{cm}")
        print(f"精度 (Accuracy): {accuracy:.4f}")
        print(f"再現率 (Recall): {recall:.4f}")
        print(f"適合率 (Precision): {precision:.4f}")
        print(f"F1スコア (F1 Score): {f1:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print(f"Thresholds:{thresholds}")

    else:
        accuracy_dict, real_index_dict, pred_index_dict, report_dict, hamming, jaccard = multilabel_evaluate(args, preds, labels, test_dataset.all_classes)

        report_df = pd.DataFrame(report_dict).T
        report_df.to_csv(os.path.join(args.experiment_path, 'report.csv'))

        for label_combination, accuracy_value in accuracy_dict.items():
            logger.info(f'{label_combination} accuracy : {accuracy_value[0]}/{accuracy_value[1]}')
            logger.info(f'{label_combination} accuracy : {accuracy_value[2]}')
        for label_combination, index_list in real_index_dict.items():
            logger.info(f'{label_combination} real_index : {index_list}')
        for label_combination, index_list in pred_index_dict.items():
            logger.info(f'{label_combination} pred_index : {index_list}')
        for label_name, scores in report_dict.items():
            logger.info(f'{label_name} : {scores}')
        all_acc = multilabel_accuracy(args, preds, labels)
        logger.info(f'all_accuracy : {all_acc}')
        logger.info(f'hamming_loss : {hamming}')
        logger.info(f'jaccard_score : {jaccard}')
