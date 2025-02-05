import glob, os, sys, pdb, time
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

import logging
logger = logging.getLogger(__name__)

def q(text = ''): # easy way to exiting the script. useful while debugging
    logger.info('> ', text)
    sys.exit()

class XRaysDataset(Dataset):
    def __init__(self, args, data_dir, transform = None, subset = 'train'):
        self.data_dir = data_dir
        self.subset = subset
        self.transform = transform
        # logger.info('self.data_dir: ', self.data_dir)

        # full dataframe including train_val and test set
        self.df = self.get_df()
        logger.info('self.df.shape: {}'.format(self.df.shape))

        if self.subset == 'train' or self.subset == 'val':            
                
            # Load or create train_val_df
            # self.train_val_df = self.load_or_create_df(
            #     args.pkl_dir_path, 
            #     args.train_val_df_pkl_path, 
            #     'self.train_val_df'
            # )
            self.train_val_df = self.get_train_val_df()

            self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()

            # if not os.path.exists(os.path.join(args.pkl_dir_path, args.disease_classes_pkl_path)):
            #     # pickle dump the classes list
            #     with open(os.path.join(args.pkl_dir_path, args.disease_classes_pkl_path), 'wb') as handle:
            #         pickle.dump(self.all_classes, handle, protocol = pickle.HIGHEST_PROTOCOL)
            #         print('\n{}: dumped'.format(args.disease_classes_pkl_path))
            # else:
            #     print('\n{}: already exists'.format(args.disease_classes_pkl_path))

            #self.new_df = self.train_val_df

            self.new_df = self.train_val_df.iloc[self.the_chosen, :] # this is the sampled train_val data
            #logger.info('\nself.all_classes_dict: {}'.format(self.all_classes_dict))

        elif self.subset == 'test':

            # Load or create test_df
            # self.new_df = self.load_or_create_df(
            #     args.pkl_dir_path, 
            #     args.test_df_pkl_path, 
            #     'self.test_df'
            # )
            self.test_df = self.get_test_df()
            self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()
            self.new_df = self.test_df.iloc[self.the_chosen, :]

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

        img = cv2.imread(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')

        # グレースケール画像（1次元）なので3次元に増やす
        if img.ndim != 3:
            img = img[:,:,np.newaxis]
            img = np.tile(img, (1, 1, 3))
        
        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1            
    
        if self.transform is not None:
            img = self.transform(img)
    
        return img, target

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

    def choose_the_indices(self):
        
        if self.subset == 'train' or self.subset == 'val':
            length = len(self.train_val_df)
        else:
            length = len(self.test_df)
        all_classes = {}
        for i in tqdm(range(length)):
            if self.subset == 'train' or self.subset == 'val':
                labels = self.train_val_df.iloc[i]['Finding Labels'].split('|')
            else:
                labels = self.test_df.iloc[i]['Finding Labels'].split('|')

            for label in labels:
                all_classes[label] = all_classes.get(label, 0) + 1
        desc_dic = sorted(all_classes.items(),key=lambda x:x[1])

        #max_examples_per_class = desc_dic[0][1]
        #max_examples_per_class = desc_dic[-2][1]
        max_examples_per_class = 10
        all_classes = {}
        the_chosen = []
        multi_count = 0
        # for i in tqdm(range(len(merged_df))):
        print('\nSampling the huuuge training dataset')
        for i in tqdm(list(np.random.choice(range(length),length, replace = False))):
            
            if self.subset == 'train' or self.subset == 'val':
                labels = self.train_val_df.iloc[i]['Finding Labels'].split('|')
            else:
                labels = self.test_df.iloc[i]['Finding Labels'].split('|')

            if 'Hernia' in labels:
                the_chosen.append(i)
                for label in labels:
                    all_classes[label] = all_classes.get(label, 0) + 1
                continue

            # if len(labels) > 1:
            #     the_chosen.append(i)
            #     multi_count += 1
            #     for label in labels:
            #         all_classes[label] = all_classes.get(label, 0) + 1
            #     continue

            if all(all_classes.get(label, 0) < max_examples_per_class for label in labels):
                the_chosen.append(i)
                for label in labels:
                    all_classes[label] = all_classes.get(label, 0) + 1

        return the_chosen, sorted(list(all_classes)), all_classes
    
    def resample(self):
        self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()
        self.new_df = self.train_val_df.iloc[self.the_chosen, :]
        logger.info('\nself.all_classes_dict: {}'.format(self.all_classes_dict))