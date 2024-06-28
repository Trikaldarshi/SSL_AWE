# evaluate the performance of pooling for Word discrimination task
#################
# step 1: get the test and valiation dataloader
# step 2: write avg_precision code with embedding exraction replaced by pooling
#################

import torch
import numpy as np
from utility_functions.utils_function import (average_precision, collate_fn)
from utility_functions.awe_dataset_class import (awe_dataset_pre_computed,
                                                 cae_awe_dataset_pre_computed)
import random
from torch.utils.data import Dataset
import pandas as pd

base_dir = ''
metadata_file='/mnt/parscratch/users/acw21am/private/MLS_Features/english/wavlm_woc/feature_metadata.csv'

def cal_precision(loader,device,distance='cosine'):
  embeddings, words = [], []
  with torch.no_grad():
    for _, (data,lens,word_name,_) in enumerate(loader):

      
      data, lens  = data.to(device), lens.to(device)
      data = torch.mean(data,dim=1)
    #   data = data/torch.norm(data,dim=1).unsqueeze(1)

      embeddings.append(data)
      words.append(word_name)
  words = np.concatenate(words)
  uwords = np.unique(words)
  word2id = {v: k for k, v in enumerate(uwords)}
  ids = [word2id[w] for w in words]
  embeddings, ids = torch.cat(embeddings,0).to(torch.float16), np.array(ids)
  avg_precision,_ = average_precision(embeddings.cpu(),ids, distance)

  return avg_precision

# For reproducibility

torch.manual_seed(3112)
torch.cuda.manual_seed(3112)
torch.cuda.manual_seed_all(3112)
np.random.seed(3112)
random.seed(3112)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(3121)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32

# print("available device:",device)
print("Is device CUDA:", device.type=="cuda")
if device.type == "cuda":
    num_workers = 4
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

print("number of workers:", num_workers)
print("pin memory status:", pin_memory)


train_data = awe_dataset_pre_computed(
    feature_df=base_dir + metadata_file,
    partition="train"
)
val_data = awe_dataset_pre_computed(
    feature_df=base_dir + metadata_file,
    partition="dev"
)
test_data = awe_dataset_pre_computed(
    feature_df=base_dir + metadata_file,
    partition="test"
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last = False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last = False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g
)



# call the function average_precision
avg_precision = cal_precision(device=device,loader=val_loader,distance='cosine')
print("average precision dev set:",avg_precision)

# call the function average_precision
avg_precision = cal_precision(device=device,loader=test_loader,distance='cosine')
print("average precision test set:",avg_precision)


print("Evluation of pooling with unseen words in dev and test set")

def split_str(path):
    return path.split("/")[-1].split("_")[0]

df = pd.read_csv(base_dir + metadata_file)


df_train = df[df["partition"]=="train"]
df_dev = df[df["partition"]=="dev"]
df_test = df[df["partition"]=="test"]

df_train = df_train.reset_index(drop=True)
df_dev = df_dev.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train['word'] = df_train['path'].apply(split_str)
df_dev['word'] = df_dev['path'].apply(split_str)
df_test['word'] = df_test['path'].apply(split_str)

# get the unique words
unique_words = df_train['word'].unique()

# filter out the words from the dev and test set that are not in the training set
df_dev = df_dev[~df_dev['word'].isin(unique_words)]
df_test = df_test[~df_test['word'].isin(unique_words)]

df_dev = df_dev.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# total number of words in the dev and test set
print('# words in dev set',len(df_dev))
print('# words in test set',len(df_test))


# print the length unique words in the dev and test set
print('# unique words in dev set',len(df_dev['word'].unique()))
print('# unique words in test set',len(df_test['word'].unique()))

class awe_dataset_pre_computed(Dataset):
    def __init__(self, feature_df):
        self.metadata = feature_df
    
        

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        SSL_feature_path = self.metadata.iloc[idx, 0]
        word_name = SSL_feature_path.split("/")[-1].split("_")[0]
        sp_id = SSL_feature_path.split("/")[-1].split("_")[3].split(".")[0]
        word_features = torch.load(base_dir + SSL_feature_path)

        return torch.squeeze(word_features),torch.tensor(word_features.size()[1]), word_name, sp_id
    
val_data = awe_dataset_pre_computed(df_dev)
test_data = awe_dataset_pre_computed(df_test)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last = False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last = False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g
)


# call the function average_precision
avg_precision = cal_precision(device=device,loader=val_loader,distance='cosine')
print("average precision dev set:",avg_precision) 

# call the function average_precision
avg_precision = cal_precision(device=device,loader=test_loader,distance='cosine')
print("average precision test set:",avg_precision)