import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os, time, random, pdb
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import pickle
import tqdm, pdb
from sklearn.metrics import roc_auc_score
from tools.evaluate import multilabel_evaluate, multiclass_evaluate, multilabel_accuracy, pred_n_write
from tools.plot import save_model

from tqdm import tqdm

from tools.plot import make_plot, get_roc_auc_score

import logging
logger = logging.getLogger(__name__)

def get_resampled_train_val_dataloaders(XRayTrain_dataset, transform, bs):
    '''
    Resamples the XRaysTrainDataset class object and returns a training and a validation dataloaders, by splitting the sampled dataset in 80-20 ratio.
    '''
    XRayTrain_dataset.resample()

    train_percentage = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset, [int(len(XRayTrain_dataset)*train_percentage), len(XRayTrain_dataset)-int(len(XRayTrain_dataset)*train_percentage)])

    logger.info('\n-----Resampled Dataset Information-----')
    logger.info('num images in train_dataset   : {}'.format(len(train_dataset)))
    logger.info('num images in val_dataset     : {}'.format(len(val_dataset)))
    logger.info('---------------------------------------')

    # make dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle = True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size = bs, shuffle = not True)

    logger.info('\n-----Resampled Batchloaders Information -----')
    logger.info('num batches in train_loader: {}'.format(len(train_loader)))
    logger.info('num batches in val_loader  : {}'.format(len(val_loader)))
    logger.info('---------------------------------------------\n')

    return train_loader, val_loader
    
# def train_epoch(device, train_loader, model, loss_fn, optimizer, epochs_till_now, final_epoch, log_interval):
#     '''
#     Takes in the data from the 'train_loader', calculates the loss over it using the 'loss_fn' 
#     and optimizes the 'model' using the 'optimizer'  
    
#     Also prints the loss and the ROC AUC score for the batches, after every 'log_interval' batches. 
#     '''
#     model.train()
    
#     running_train_loss = 0
#     train_loss_list = []

#     start_time = time.time()
#     for batch_idx, (img, target) in enumerate(train_loader):
#         # logger.info(type(img), img.shape) # , np.unique(img))

#         img = img.to(device)
#         target = target.to(device)
        
#         optimizer.zero_grad()    
#         out = model(img)        
#         loss = loss_fn(out, target)
#         running_train_loss += loss.item()*img.shape[0]
#         train_loss_list.append(loss.item())

#         loss.backward()
#         optimizer.step()
        
#         if (batch_idx+1)%log_interval == 0:
            
#             batch_time = time.time() - start_time
#             m, s = divmod(batch_time, 60)
#             logger.info('Train Loss for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(str(batch_idx+1).zfill(3), str(len(train_loader)).zfill(3), epochs_till_now, final_epoch, round(loss.item(), 5), int(m), round(s, 2)))
        
#         start_time = time.time()
            
#     return train_loss_list, running_train_loss/float(len(train_loader.dataset))

# def val_epoch(args, device, val_loader, model, loss_fn, epochs_till_now = None, final_epoch = None, log_interval = 1):
#     '''
#     It essentially takes in the val_loader/test_loader, the model and the loss function and evaluates
#     the loss and the ROC AUC score for all the data in the dataloader.
    
#     It also prints the loss and the ROC AUC score for every 'log_interval'th batch, only when 'test_only' is False
#     '''
#     model.eval()

#     running_val_loss = 0
#     val_loss_list = []
#     val_loader_examples_num = len(val_loader.dataset)

#     probs = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
#     gt    = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
#     k=0

#     with torch.no_grad():
#         batch_start_time = time.time()    
#         for batch_idx, (img, target) in enumerate(val_loader):
            
#             img = img.to(device)
#             target = target.to(device)    
    
#             out = model(img)        
#             loss = loss_fn(out, target)    
#             running_val_loss += loss.item()*img.shape[0]
#             val_loss_list.append(loss.item())

#             # storing model predictions for metric evaluat`ion 
#             probs[k: k + out.shape[0], :] = out.cpu()
#             gt[   k: k + out.shape[0], :] = target.cpu()
#             k += out.shape[0]

#             if ((batch_idx+1)%log_interval == 0): # only when ((batch_idx + 1) is divisible by log_interval) and (when test_only = False)
#                 # batch metric evaluation
# #                 batch_roc_auc_score = get_roc_auc_score(target, out)

#                 batch_time = time.time() - batch_start_time
#                 m, s = divmod(batch_time, 60)
#                 logger.info('Val Loss   for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(str(batch_idx+1).zfill(3), str(len(val_loader)).zfill(3), epochs_till_now, final_epoch, round(loss.item(), 5), int(m), round(s, 2)))
            
#             batch_start_time = time.time()    
            
#     # metric scenes
#     roc_auc = get_roc_auc_score(args, gt, probs)

#     return val_loss_list, running_val_loss/float(len(val_loader.dataset)), roc_auc

# モデル訓練と評価の関数
def train_epoch(model, train_loader, optimizer, criterion, device='cuda'):

    model.train()
    train_batch_loss = []
    
    for image, label in train_loader:
        
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()

            output = model(image)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            train_batch_loss.append(loss.item())
        
    return train_batch_loss, np.mean(train_batch_loss)

def val_epoch(args, model, val_loader, criterion, device='cuda'):
    
    model.eval()
    valid_batch_loss = []

    val_loader_examples_num = len(val_loader.dataset)

    probs = np.zeros((val_loader_examples_num, args.class_numbers), dtype = np.float32)
    gt    = np.zeros((val_loader_examples_num, args.class_numbers), dtype = np.float32)
    k=0

    with torch.no_grad():
        for image, label in val_loader:
                
                image = image.to(device)
                label = label.to(device)

                output = model(image)

                loss   = criterion(output, label)

                valid_batch_loss.append(loss.item())

                # storing model predictions for metric evaluat`ion 
                probs[k: k + output.shape[0], :] = output.cpu()
                gt[   k: k + output.shape[0], :] = label.cpu()
                k += output.shape[0]

    # metric scenes
    roc_auc = get_roc_auc_score(args, gt, probs)

    return valid_batch_loss, np.mean(valid_batch_loss), roc_auc


def fit(args, device, train_loader, val_loader, model, criterion, optimizer):

    #logger.info(f'======= Training after epoch #{epochs_till_now}... =======')

    epoch_train_loss = []
    epoch_val_loss = []
    total_train_loss_list = []
    total_val_loss_list = []

    best_info = {
        'epoch': 0,
        'train_loss': np.inf,
        'valid_loss': np.inf
    }

    patience = 5  # 例えば5エポックで改善がなければ停止
    patience_counter = 0

    #since = time.time()
    epoch_start_time = time.time()
    logger.info("Training started.")

    epochs = args.epochs
    for epoch in tqdm(range(epochs)):
        logger.info(f'============ EPOCH {epoch+1}/{epochs} ============')
        
        
        logger.info('TRAINING')
        train_loss, mean_running_train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        logger.info('VALIDATION')
        val_loss, mean_running_val_loss, roc_auc = val_epoch(args, model, val_loader, criterion, device)

        epoch_train_loss.append(mean_running_train_loss)
        epoch_val_loss.append(mean_running_val_loss)
        total_train_loss_list.extend(train_loss)
        total_val_loss_list.extend(val_loss)

        # Early stopping & best model saving
        if mean_running_val_loss < best_info['valid_loss']:
            best_info['epoch'] = epoch+1
            best_info['train_loss'] = mean_running_train_loss
            best_info['valid_loss'] = mean_running_val_loss

            #save_path = os.path.join(args.experiment_path, 'best.pth')
            save_model(model, optimizer, epoch+1, best_info['epoch'], best_info['valid_loss'], {
                'epoch_train_loss': epoch_train_loss,
                'epoch_val_loss': epoch_val_loss,
                'total_train_loss_list': total_train_loss_list,
                'total_val_loss_list': total_val_loss_list
            }, os.path.join(args.experiment_path, 'best.pth'))
            patience_counter = 0  
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info('Early stopping triggered!')
                break

        # ログ出力 (10エポックごと & ベスト更新時)
        if ((epoch+1) % 10 == 0) or (mean_running_val_loss < best_info['valid_loss']):
            logger.info(f'epoch: {epoch+1}/{epochs}, train loss: {mean_running_train_loss:.4f}, valid loss: {mean_running_val_loss:.4f}')

    make_plot(epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list, os.path.join(args.experiment_path, 'losses.png'))
    print('loss plots saved !!!')

    # エポックごとの時間計測
    total_epoch_time = time.time() - epoch_start_time
    m, s = divmod(total_epoch_time, 60)
    h, m = divmod(m, 60)
    logger.info(f'Epoch {epoch+1}/{epochs} took {int(h)}h {int(m)}m {int(s)}s')

def test(args, model, test_loader, device):

    logger.info('\n======= Testing... =======\n')
    model.eval()

    predictions = []
    labels      = []
    all_test_data = []

    logger.info('start test')
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            all_test_data.append(image.cpu().numpy())

            predictions.append(torch.sigmoid(output))
            labels.append(label)

    logger.info('end test')

    predictions = torch.cat(predictions, axis=0)
    labels = torch.cat(labels, axis=0)

    predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    #pred_n_write(args, test_loader, model, 'test')

    return predictions, labels