import itertools as it
import os

import numpy as np
import pandas as pd
import torch
from scipy.special import comb
from torch.utils.data import Dataset

from utility_functions.feature_extractor import (SSL_features, clip_features,
                                                 load_model)

base_dir = ''

def split_string(strs):
    return strs.split(sep="/")[-1].split("_")[0]

class awe_dataset_pre_computed(Dataset):
    def __init__(self, feature_df, partition):
        self.metadata = pd.read_csv(feature_df)
        self.partition = partition
        self.metadata = self.metadata[self.metadata["partition"]==self.partition]
    
        

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        SSL_feature_path = self.metadata.iloc[idx, 0]
        word_name = SSL_feature_path.split("/")[-1].split("_")[0]
        sp_id = SSL_feature_path.split("/")[-1].split("_")[3].split(".")[0]
        word_features = torch.load(base_dir + SSL_feature_path)

        return torch.squeeze(word_features),torch.tensor(word_features.size()[1]), word_name, sp_id
    
# For correspondece autoencoder 
# use only for train loader that's it.
class cae_awe_dataset_pre_computed(Dataset):
    def __init__(self, feature_df, partition):
        self.metadata = pd.read_csv(feature_df)
        self.partition = partition
        self.metadata = self.metadata[self.metadata["partition"]==self.partition]

        if self.partition=="train":
            self.metadata_copy = self.metadata.copy()
            self.metadata_copy["word_name"] = self.metadata_copy["path"].apply(split_string)
            self.x_idx, self.y_idx  = np.arange(len(self.metadata_copy)), np.arange(len(self.metadata_copy))
            labels = self.metadata_copy["word_name"].values
            num_examples = len(labels)
            num_pairs = int(comb(num_examples, 2))
            # build up binary array of matching examples
            matches = np.zeros(num_pairs, dtype= bool)
            i = 0
            for n in range(num_examples):
                j = i + num_examples - n - 1
                matches[i:j] = (labels[n] == labels[n + 1:]).astype(np.int32)
                i = j
            num_same = np.sum(matches)
            pairs_numerical = it.combinations(np.arange(len(labels)), 2)
            matched_pairs = []
            for i, pair in enumerate(pairs_numerical):
                if matches[i]:
                    matched_pairs.append(pair)
            matched_pairs = np.array(matched_pairs)  
            #if condition returns True, then nothing happens:
            assert len(matched_pairs)==num_same
            self.x_idx = np.concatenate((self.x_idx, matched_pairs[:,0]))
            #self.x_idx = np.concatenate((self.x_idx, matched_pairs[:,1]))
            self.y_idx = np.concatenate((self.y_idx, matched_pairs[:,1]))
            #self.y_idx = np.concatenate((self.y_idx, matched_pairs[:,0]))
    
        

    def __len__(self):
        return len(self.x_idx)


    def __getitem__(self, idx):
        SSL_feature_path_x = self.metadata.iloc[self.x_idx[idx], 0]
        SSL_feature_path_y = self.metadata.iloc[self.y_idx[idx], 0]

        word_name_x = SSL_feature_path_x.split("/")[-1].split("_")[0]
        word_name_y = SSL_feature_path_y.split("/")[-1].split("_")[0]
        assert word_name_x==word_name_y
        sp_id_x = SSL_feature_path_x.split("/")[-1].split("_")[3].split(".")[0]
        sp_id_y = SSL_feature_path_y.split("/")[-1].split("_")[3].split(".")[0]

        word_features_x = torch.load(SSL_feature_path_x)
        word_features_y = torch.load(SSL_feature_path_y)

        return torch.squeeze(word_features_x),torch.tensor(word_features_x.size()[1]), word_name_x, sp_id_x, \
            torch.squeeze(word_features_y),torch.tensor(word_features_y.size()[1]), word_name_y, sp_id_y
