from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import hamming_loss, jaccard_score, multilabel_confusion_matrix, classification_report

from collections import defaultdict
import sys, os, time, random, pdb
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import pickle
import tqdm, pdb

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

def pred_n_write(args, test_loader, model, save_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    # model.eval()  # ループの外に移動

    # num_samples = len(test_loader.dataset)
    # num_classes = 15  # クラス数は動的に変更可能なら変える

    # res = np.zeros((num_samples, num_classes), dtype=np.float32)

    # labels = []
    # k = 0
    # for batch_idx, img, label in tqdm(enumerate(test_loader)):
    #     img = img.to(device)  # GPU に送る
    #     with torch.no_grad():
    #         pred = torch.sigmoid(model(img)).cpu().numpy()  # CPU に戻して NumPy 配列化
    #         res[k: k + pred.shape[0], :] = pred
    #         k += pred.shape[0]
    #         labels.append(label)
    # labels = labels.cpu().detach().numpy()

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

    th = args.threshold

    preds_array = np.array(predictions)
    preds_list = np.where(preds_array > th, 1.0, 0.0).tolist()

    # write csv
    logger.info('populating the csv')
    submit = pd.DataFrame()
    submit['ImageID'] = [str.split(i, os.sep)[-1] for i in test_loader.dataset.data_list]
    with open('pickeles/disease_classes.pickle', 'rb') as handle:
        disease_classes = pickle.load(handle)

    for idx, col in enumerate(disease_classes):
        if col == 'Hernia':
            submit['Hern'] = preds_list[:, idx]
        elif col == 'Pleural_Thickening':
            submit['Pleural_thickening'] = preds_list[:, idx]
        elif col == 'No Finding':
            submit['No_findings'] = preds_list[:, idx]
        else:
            submit[col] = preds_list[:, idx]
    rand_num = str(random.randint(1000, 9999))
    csv_name = '{}___{}.csv'.format(save_name, rand_num)
    submit.to_csv('res/' + csv_name, index = False)
    logger.info('{} saved !'.format(csv_name))

def evaluate_all(preds, labels):
    """評価指標での評価

    Args:
        preds (list): 予測結果
        labels (list): ラベル

    Returns:
        dict : 評価結果 
        (混合行列,精度,適合率,再現率,F1スコア)
    """

    cm        = confusion_matrix(labels, preds)
    acc       = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall    = recall_score(labels, preds, average='macro')
    f1        = f1_score(labels, preds, average='macro')
    
    return cm, acc, precision, recall, f1

def binary_evaluate(preds, labels):
    """2値分類での評価

    Args:
        preds (list): 予測結果
        labels (list): ラベル

    Returns:
        dict : 評価結果 
        (混合行列,精度,適合率,再現率,F1スコア)
    """
    
    binary_preds  = [1 if x > 0 else 0 for x in preds]
    binary_labels = [1 if x > 0 else 0 for x in labels]

    cm        = confusion_matrix(binary_labels, binary_preds)
    acc       = accuracy_score(binary_labels, binary_preds)
    precision = precision_score(binary_labels, binary_preds)
    recall    = recall_score(binary_labels, binary_preds)
    f1        = f1_score(binary_labels, binary_preds)

    return cm, acc, precision, recall, f1

def multiclass_evaluate(args, preds, labels, branch_types):
    """マルチクラス分類での評価

    Args:
        preds (list): 予測結果
        labels (list): ラベル

    Returns:
        dict : 評価結果 
        (実患者数, 実患者番号, 予測患者番号, 正解数, 精度)
    """
    
    # preds の各予測のインデックスに基づいて one-hot エンコード
    preds_list = np.eye(len(branch_types))[np.argmax(preds, axis=1)].tolist()

    # 各ラベルのワンホットベクトルをあらかじめ生成
    one_hot_labels = [np.eye(len(branch_types))[j] for j in range(len(branch_types))]

    # 結果を格納する辞書を準備
    results = {branch: {'patient_count': 0, 'true_patient_index': [], 'pred_patient_index': [], 'matched_patient': 0} 
               for branch in branch_types}

    # 各ラベルごとにループ
    for j, label_name in enumerate(branch_types):
        l = one_hot_labels[j]

        # 予測と実際のラベルを比較
        for idx, (real, pred) in enumerate(zip(labels, preds_list)):
            # 実際のラベルが該当ラベルに一致するかチェック
            if np.array_equal(real, l):
                results[label_name]['patient_count'] += 1
                results[label_name]['true_patient_index'].append(idx)

                # 予測も該当ラベルに一致する場合
                if np.array_equal(pred, l):
                    results[label_name]['matched_patient'] += 1
                    results[label_name]['pred_patient_index'].append(idx)

            # 実際のラベルは一致しないが予測が一致する場合
            elif np.array_equal(pred, l):
                results[label_name]['pred_patient_index'].append(idx)

    # 結果の表示
    for label_name, data in results.items():
        patient_count = data['patient_count']
        matched_patient = data['matched_patient']
        accuracy = float(matched_patient) / patient_count if patient_count > 0 else 0.0

        results[label_name]["accuracy"] = accuracy
        
    report_dict = classification_report(labels, preds_list ,target_names=branch_types, output_dict=True, zero_division=0)

    return results, report_dict


def multilabel_evaluate(args, preds, labels, branch_types):
    """マルチラベル分類での評価

    Args:
        preds (list): 予測結果
        labels (list): ラベル

    Returns:
        dict : 評価結果 
        (精度，本患者, 予測患者, 分類レポート, ハミング距離, ジャッカードスコア)
    """
    
    th = args.threshold

    preds_array = np.array(preds)
    preds_list = np.where(preds_array > th, 1.0, 0.0).tolist()

    arr_preds = np.array(preds_list)
    
    # 各ラベル組み合わせに対応するカウントと一致カウントを保持する辞書
    count_dict = defaultdict(int)
    matched_dict = defaultdict(int)
    real_index_dict = defaultdict(list)
    pred_index_dict = defaultdict(list)
    accuracy_dict = defaultdict(list)

    count = 0

    # ラベルごとのマッピングをクラス名に変換
    for real, pred in zip(labels, preds_list):
        real_classes = frozenset([branch_types[i] for i, val in enumerate(real) if val == 1])
        pred_classes = frozenset([branch_types[i] for i, val in enumerate(pred) if val == 1])

        # 正解ラベルのカウントとインデックスを更新
        count_dict[real_classes] += 1
        real_index_dict[real_classes].append(count)
        pred_index_dict[pred_classes].append(count)

        # 一致する場合、一致カウントを更新
        if real_classes == pred_classes:
            matched_dict[real_classes] += 1

        count += 1

    # 各組み合わせの正解率を表示
    for label_combination, count_value in count_dict.items():
        if count_value != 0:
            matched_value = matched_dict[label_combination]

            accuracy = float(matched_value) / float(count_value)
            
            accuracy_dict[label_combination].append(matched_value)
            accuracy_dict[label_combination].append(count_value)
            accuracy_dict[label_combination].append(accuracy)
    
    
    report_dict = classification_report(labels, preds_list, target_names=branch_types, output_dict=True, zero_division=0)
    hamming = hamming_loss(labels, preds_list)
    jaccard = jaccard_score(labels, preds_list, average=None)

    return accuracy_dict, real_index_dict, pred_index_dict, report_dict, hamming, jaccard

def multilabel_accuracy(args, preds, labels):
    """
    マルチラベル分類での精度

    Args:
        preds (list): 予測結果
        labels (list): ラベル

    Returns:
        all_accuracy: 評価結果
    """

    th = args.threshold

    preds_array = np.array(preds)
    preds_list = np.where(preds_array > th, 1.0, 0.0).tolist()

    arr_preds = np.array(preds_list)
    
    total = 0
    matched = 0

    # 各データポイントごとに一致度を確認
    for real, pred in zip(labels, arr_preds):
        pred = list(pred)
        real = list(real)
        
        # ラベル組み合わせの総数をカウント
        total += 1

        # 一致していれば正解数を増加
        if real == pred:
            matched += 1
    
    all_accuracy = matched / total

    return all_accuracy