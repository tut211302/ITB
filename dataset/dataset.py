import glob, os, sys, pdb, time
import pandas as pd
import numpy as np
import cv2
import pickle
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

import logging
logger = logging.getLogger(__name__)

def q(text = ''): # easy way to exiting the script. useful while debugging
    logger.info('> ', text)
    sys.exit()

def apply_CLAHE(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    CLAHE を適用する関数
    :param image: 入力画像 (numpy array)
    :param clip_limit: コントラスト制限
    :param grid_size: タイルグリッドサイズ
    :return: CLAHE 適用後の画像
    """

    # BGR チャンネルを分割
    b, g, r = cv2.split(image)

    # CLAHE の設定
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 各チャンネルに適用
    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    r_clahe = clahe.apply(r)

    # チャンネルを統合
    image_clahe = cv2.merge([b_clahe, g_clahe, r_clahe])

    #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # グレースケール変換
    #clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    #image_clahe = clahe.apply(image_gray)
    return image_clahe

class XRaysDataset(Dataset):
    def __init__(self, args, data_dir, class_numbers, chosen_class = None, transform = None, subset = 'train'):
        self.data_dir = data_dir
        self.subset = subset
        self.transform = transform
        self.class_numbers = class_numbers
        self.chosen_class = chosen_class
        # logger.info('self.data_dir: ', self.data_dir)

        # full dataframe including train_val and test set
        self.df = self.get_df()
        logger.info('self.df.shape: {}'.format(self.df.shape))

        if self.subset == 'train' or self.subset == 'val':            
                
            # Load or create train_val_df
            self.train_val_df = self.load_or_create_df(
                args.pkl_dir_path, 
                args.train_val_df_pkl_path, 
                'self.train_val_df'
            )
            #self.train_val_df = self.get_train_val_df()

            self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()

            # if not os.path.exists(os.path.join(args.pkl_dir_path, args.disease_classes_pkl_path)):
            #     # pickle dump the classes list
            with open(os.path.join(args.pkl_dir_path, args.disease_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_classes, handle, protocol = pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(args.disease_classes_pkl_path))
            # else:
            #     print('\n{}: already exists'.format(args.disease_classes_pkl_path))

            #self.new_df = self.train_val_df

            self.new_df = self.train_val_df.iloc[self.the_chosen, :] # this is the sampled train_val data
            #logger.info('\nself.all_classes_dict: {}'.format(self.all_classes_dict))

        elif self.subset == 'test':

            #Load or create test_df
            self.new_df = self.load_or_create_df(
                args.pkl_dir_path, 
                args.test_df_pkl_path, 
                'self.test_df'
            )
            #self.new_df = self.get_test_df()
            #self.new_df = self.new_df.iloc[50]

            # loading the classes list
            with open(os.path.join(args.pkl_dir_path, args.disease_classes_pkl_path), 'rb') as handle:
                self.all_classes = pickle.load(handle)
            

    def load_or_create_df(self, pkl_dir_path, pkl_file_path, df_name):
        """
        Load a dataframe from a pickle file if it exists, otherwise create it using the provided method
        and save it to the pickle file.

        Parameters:
            get_df_method (function): Function to create the dataframe if the pickle file doesn't exist.
            pkl_dir_path (str): Path to the directory containing the pickle file.
            pkl_file_path (str): Path to the pickle file.
            df_name (str): Name of the dataframe (for logging purposes).

        Returns:
            pd.DataFrame: Loaded or newly created dataframe.
        """
        full_path = os.path.join(pkl_dir_path, pkl_file_path)

        if not os.path.exists(full_path):
            if df_name == 'self.train_val_df':
                df = self.get_train_val_df()
            else:
                df = self.get_test_df()
            print(f'\n{df_name}.shape: {df.shape}')

            # Save the dataframe to a pickle file
            with open(full_path, 'wb') as handle:
                pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'{pkl_file_path}: dumped')
        else:
            # Load the dataframe from the pickle file
            with open(full_path, 'rb') as handle:
                df = pickle.load(handle)
            print(f'\n{pkl_file_path}: loaded')
            print(f'{df_name}.shape: {df.shape}')

        return df

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, index):
        row = self.new_df.iloc[index, :]

        #img = row['image_links']
        img = cv2.imread(row['image_links'])
        #img = self.load_image(row['image_links'])
        img = apply_CLAHE(img)
        
        labels = str.split(row['Finding Labels'], '|')
        
        if img.ndim != 3:
           img = img[:,:,np.newaxis]
           img = np.tile(img, (1, 1, 3))
        
        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            if lab in self.all_classes:
                lab_idx = self.all_classes.index(lab)
                target[lab_idx] = 1            
    
        if self.transform is not None:
            img = self.transform(img)
    
        return img, target
    
    def load_image(self, path):
        image = Image.open(path)
        return image

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        logger.info('\n{} found: {}'.format(csv_path, os.path.exists(csv_path)))
        
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]

        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df

    def get_train_val_list(self):
        f = open(os.path.join(self.data_dir, 'train_val_list.txt'), 'r')
        train_val_list = str.split(f.read(), '\n')
        return train_val_list

    def get_train_val_df(self):

        # get the list of train_val data 
        train_val_list = self.get_train_val_list()

        train_val_df = pd.DataFrame()
        logger.info('\nbuilding train_val_df...')
        train_val_df = self.df[self.df.iloc[:, 0].apply(lambda x: os.path.basename(x) in train_val_list)].copy()

        # logger.info('train_val_df.shape: {}'.format(train_val_df.shape))

        return train_val_df
    
    def get_test_list(self):
        f = open( os.path.join(self.data_dir, 'test_list.txt'), 'r')
        test_list = str.split(f.read(), '\n')
        return test_list

    def get_test_df(self):

        # get the list of test data 
        test_list = self.get_test_list()

        test_df = pd.DataFrame()
        logger.info('\nbuilding test_df...')
        test_df = self.df[self.df.iloc[:, 0].apply(lambda x: os.path.basename(x) in test_list)].copy()
         
        #logger.info('test_df.shape: ', test_df.shape)

        return test_df
    
    def find_min_key(keys_list, data_dict):
    # キーリスト内のキーのうち、最小値を持つキーを探す
        return min(keys_list, key=lambda k: data_dict.get(k, float('inf')))

    def choose_the_indices(self):
        
        if self.subset == 'train' or self.subset == 'val':
            length = len(self.train_val_df)
        else:
            length = len(self.test_df)
        all_classes = {}
        for i in tqdm(range(length)):
            if self.subset == 'train' or self.subset == 'val':
                labels = self.train_val_df.iloc[i]['Finding Labels'].split('|')

            for label in labels:
                all_classes[label] = all_classes.get(label, 0) + 1
        desc_dic = sorted(all_classes.items(),key=lambda x:x[1])
        asc_dic = sorted(all_classes.items(), key=lambda x: x[1], reverse=True)

        if self.chosen_class == None:
            self.chosen_class = [label for label, _ in asc_dic][1:self.class_numbers+1]
            max_examples_per_class = 1000000
        else:
            #min_key = self.find_min_key(self.chosen_class, all_classes)
            self.chosen_class = [label for label, _ in asc_dic][1:self.class_numbers+1]
            max_examples_per_class = all_classes[self.chosen_class[-1]]

        #max_examples_per_class = desc_dic[0][1]
        all_classes = {}
        the_chosen = []
        multi_count = 0
        no_finding_count = 0
        # for i in tqdm(range(len(merged_df))):
        print('\nSampling the huuuge training dataset')
        for i in tqdm(list(np.random.choice(range(length),length, replace = False))):
            
            if self.subset == 'train' or self.subset == 'val':
                labels = self.train_val_df.iloc[i]['Finding Labels'].split('|')

            if 'No Finding' in labels and self.class_numbers == 1:
                if no_finding_count < max_examples_per_class:
                    the_chosen.append(i)
                    no_finding_count += 1
                    # for label in labels:
                    #     all_classes[label] = all_classes.get(label, 0) + 1
                    continue
            
            # if any(x in labels for x in chosen_class):
            if any(x in self.chosen_class for x in labels):
                if all(all_classes.get(label, 0) < max_examples_per_class for label in labels):
                    the_chosen.append(i)
                    for label in labels:
                        if label in self.chosen_class:
                            all_classes[label] = all_classes.get(label, 0) + 1

        
            # if 'Hernia' in labels:
            #     the_chosen.append(i)
            #     for label in labels:
            #         all_classes[label] = all_classes.get(label, 0) + 1
            #     continue

            # if len(labels) > 1:
            #     the_chosen.append(i)
            #     multi_count += 1
            #     for label in labels:
            #         all_classes[label] = all_classes.get(label, 0) + 1
            #     continue

            # if all(all_classes.get(label, 0) < max_examples_per_class for label in labels):
            #     the_chosen.append(i)
            #     for label in labels:
            #         all_classes[label] = all_classes.get(label, 0) + 1

        return the_chosen, self.chosen_class, all_classes
    
    def resample(self):
        self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()
        self.new_df = self.train_val_df.iloc[self.the_chosen, :]
        logger.info('\nself.all_classes_dict: {}'.format(self.all_classes_dict))