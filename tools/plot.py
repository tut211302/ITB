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

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
from matplotlib import font_manager

import logging

logger = logging.getLogger(__name__)

def get_roc_auc_score(args, y_true, y_probs):
    '''
    Uses roc_auc_score function from sklearn.metrics to calculate the micro ROC AUC score for a given y_true and y_probs.
    '''

    with open(os.path.join(args.pkl_dir_path, args.disease_classes_pkl_path), 'rb') as handle:
        all_classes = pickle.load(handle)
    
    # Check if 'No Finding' exists and get its index
    NoFindingIndex = -1  # Default to -1 if 'No Finding' does not exist
    if 'No Finding' in all_classes:
        NoFindingIndex = all_classes.index('No Finding')

    if True:
        print('\nNoFindingIndex: ', NoFindingIndex)
        print('y_true.shape, y_probs.shape ', y_true.shape, y_probs.shape)
        GT_and_probs = {'y_true': y_true, 'y_probs': y_probs}
        with open('GT_and_probs', 'wb') as handle:
            pickle.dump(GT_and_probs, handle, protocol = pickle.HIGHEST_PROTOCOL)

    class_roc_auc_list = []    
    useful_classes_roc_auc_list = []
    
    for i in range(y_true.shape[1]):
        class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        class_roc_auc_list.append(class_roc_auc)
        if i != NoFindingIndex:
            useful_classes_roc_auc_list.append(class_roc_auc)
    # if True:
    #     print('\nclass_roc_auc_list: ', class_roc_auc_list)
    #     print('\nuseful_classes_roc_auc_list', useful_classes_roc_auc_list)

    return np.mean(np.array(useful_classes_roc_auc_list))

def make_plot(epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list, save_name):
    '''
    This function makes the following 4 different plots-
    1. mean train loss VS number of epochs
    2. mean val   loss VS number of epochs
    3. batch train loss for all the training   batches VS number of batches
    4. batch val   loss for all the validation batches VS number of batches
    '''
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('loss trends', fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('epoch train loss VS #epochs')
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('epoch train loss')
    ax1.plot(epoch_train_loss)

    ax2.title.set_text('epoch val loss VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('epoch val loss')
    ax2.plot(epoch_val_loss)

    ax3.title.set_text('batch train loss VS #batches')
    ax3.set_xlabel('#batches')
    ax3.set_ylabel('batch train loss')
    ax3.plot(total_train_loss_list)

    ax4.title.set_text('batch val loss VS #batches')
    ax4.set_xlabel('#batches')
    ax4.set_ylabel('batch val loss')
    ax4.plot(total_val_loss_list)
    
    #plt.savefig(os.path.join(args.experiment_path,'losses_{}.png'.format(save_name)))
    plt.savefig(save_name)


def save_model(model, optimizer, epoch, epoch_best, best_loss, losses_dict, save_path):
    """モデルの保存"""
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_best': epoch_best,
        'best_loss': best_loss,
        'losses_dict': losses_dict
    }, save_path)

    logger.info('Saved model at {}'.format(save_path))

# def save_loss(args, train_loss, valid_loss, experiment_path):
#     """損失の保存

#     Args:
#         args (_type_): パラメータの設定された引数
#         train_loss (_type_): エポックごとの訓練の損失
#         test_loss (_type_): エポックごとのテスト（バリデーション）の損失
#     """
#     np.save(os.path.join(experiment_path, 'trainloss'), train_loss)
#     np.save(os.path.join(experiment_path, 'valloss'), valid_loss)

#     logger.info('saved loss')

def save_train_loss_graph(args, train_loss, valid_loss, experiment_path):
    """損失グラフを保存する

    Args:
        args (_type_): _description_
        train_loss (_type_): エポックごとの訓練の損失
        valid_loss (_type_): エポックごとのテスト（バリデーション）の損失
    """
    # 日本語fontの設定（IPAexフォントの設定）
    # For Windows font_path='C:/Windows/fonts/ipaexg.ttf' 
    font_path ='/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf'
    font_prop = font_manager.FontProperties(fname=font_path)
    font_prop.set_style('normal')
    font_prop.set_weight('light')
    font_prop.set_size('12')
    fp2 = font_prop.copy()
    fp2.set_size('25')
    fp3 = font_prop.copy()
    fp3.set_size('15')

    plt.figure(figsize=(14,10))

    plt.plot(train_loss, 
            color='b', 
            linestyle='-', 
            linewidth=3, 
            path_effects=[path_effects.SimpleLineShadow(),
                        path_effects.Normal()])
    plt.plot(valid_loss, 
            color='r', 
            linestyle='--',
            linewidth=3,
            path_effects=[path_effects.SimpleLineShadow(),
                        path_effects.Normal()])

    plt.tick_params(labelsize=18)
    plt.title('Epoch-Loss Graph-'+ f'{args.model_name}',
        fontsize=30,font_properties=fp2)
    plt.ylabel('Loss',fontsize=25, font_properties=fp2)
    plt.xlabel('Epoch',fontsize=25, font_properties=fp2)
    plt.legend(['Training', 'Validation'], loc='best', fontsize=25, prop=fp2)


    plt.savefig(os.path.join(experiment_path, f'{args.model_name}_EpochLoss_graph.pdf'))
    plt.savefig(os.path.join(experiment_path, f'{args.model_name}_EpochLoss_graph.jpg'))

    logger.info('saved training graph')