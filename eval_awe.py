import torch
import numpy as np
from utility_functions.awe_dataset_class import awe_dataset_pre_computed
from utility_functions.utils_function import average_precision, collate_fn
import torch.nn as nn
import random
from utility_functions.model_cae import model_cae
import pandas as pd
import argparse
import sys
import os
from torch.utils.data import Dataset
from math import comb   


possible_models = ["HUBERT_BASE","HUBERT_LARGE","HUBERT_XLARGE","WAV2VEC2_BASE","WAV2VEC2_LARGE",
                    "WAV2VEC2_LARGE_LV60K","WAV2VEC2_XLSR53","HUBERT_ASR_LARGE","HUBERT_ASR_XLARGE",
                    "WAV2VEC2_ASR_BASE_10M","WAV2VEC2_ASR_BASE_100H","WAV2VEC2_ASR_BASE_960H",
                    "WAV2VEC2_ASR_LARGE_10M","WAV2VEC2_ASR_LARGE_100H","WAV2VEC2_ASR_LARGE_960H",
                    "WAV2VEC2_ASR_LARGE_LV60K_10M","WAV2VEC2_ASR_LARGE_LV60K_100H","WAV2VEC2_ASR_LARGE_LV60K_960H","MFCC", "SPEC",'WAVLM_BASE']
# Utility function
def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--model_name", type=str, help = "name of the model for example, HUBERT_BASE", nargs='?', default = "HUBERT_BASE", choices = possible_models)
    parser.add_argument("--input_dim", type = int, help = "dimension of input features", nargs='?', default=768)
    parser.add_argument("--metadata_file", type = str, help = "a text file or dataframe containing paths of wave files, words, start point, duration \
      or SSL features metadata file")
    parser.add_argument("--model_weights", type = str, help = "path of the pre-trained model which will be used as a embedding extractor")
    parser.add_argument("--batch_size", type = int, help = "batch_size", nargs='?', default=2)
    parser.add_argument("--embedding_dim", type = int, help = "value of embedding dimensions",nargs='?',default = 128)
    parser.add_argument("--hidden_dim", type = int, help = "rnn hidden dimension values", default=512)
    parser.add_argument("--distance", type = str, help = "type of distance to compute the similarity",nargs='?',default = "cosine")
    parser.add_argument("--rnn_type", type = str, help = " type or rnn, gru or lstm?", default="LSTM", choices=["GRU","LSTM"])
    parser.add_argument("--bidirectional", type = bool, help = " bidirectional rnn or not", default = True)
    parser.add_argument("--num_layers", type = int, help = " number of layers in rnn network, input more than 1", default=2) 
    parser.add_argument("--dropout", type = float, help = "dropout applied inside rnn network", default=0.2)

    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])

def cal_precision(model, loader, device, distance):
    embeddings, words, unique_id = [], [], []
    model = model.eval()
    with torch.no_grad():
        for idx, (data, lens, word_name, sp_ch_id) in enumerate(loader):
            lens, perm_idx = lens.sort(0, descending=True)
            data = data[perm_idx]
            word_name = word_name[perm_idx]

            data, lens  = data.to(device), lens.to(device)
            _,emb = model.encoder(data, lens)
            embeddings.append(emb)
            words.append(word_name)
            unique_id.append(sp_ch_id)
            # print(idx)
    words = np.concatenate(words)
    unique_id = np.concatenate(unique_id)
    uwords = np.unique(words)
    word2id = {v: k for k, v in enumerate(uwords)}
    ids = [word2id[w] for w in words]
    embeddings, ids = torch.cat(embeddings,0).detach().cpu(), np.array(ids)
    if len(embeddings)>=10000:
        avg_precision = 0
    else:
        avg_precision,_ = average_precision(embeddings,ids, distance, show_plot=False)
    return avg_precision, embeddings, words, unique_id, ids




# MAIN Function

def main():
    # For reproducibility
    print("This is main program")
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

    args = check_argv()



    input_dim =  args.input_dim         #768
    hidden_dim = args.hidden_dim        #256
    embedding_dim = args.embedding_dim  #128
    rnn_type = args.rnn_type            #"GRU"
    bidirectional = args.bidirectional       #True
    num_layers = args.num_layers        #4
    dropout = args.dropout              #0.2
    batch_size = args.batch_size        #64



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Is device CUDA?:", device.type=="cuda")
    if device.type == "cuda":
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    print("using pre-computed features", args.model_name)

    val_data = awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="dev"
    )
    test_data = awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="test"
    )

    print("length of validation data:",len(val_data))
    print("length of test data:",len(test_data))
    
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


    model = model_cae(input_dim, hidden_dim, embedding_dim, rnn_type,
    bidirectional, num_layers, dropout)

    checkpoint = torch.load(args.model_weights, map_location=torch.device(device))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)


    print("extracting embedding for val set")
    val_avg_precision, _, _, _, _ = cal_precision(model, val_loader, device, args.distance)
    print("val average precision:", val_avg_precision)


    print("extracting embedding for test set")
    test_avg_precision, _, _, _, _ = cal_precision(model, test_loader, device, args.distance)
    print("test average precision:", test_avg_precision)




    print("Evluation of CAE with unseen words in dev and test set")
    base_dir = ''
    def split_str(path):
        return path.split("/")[-1].split("_")[0]

    df = pd.read_csv(args.metadata_file)


    df_train = df[df["partition"]=="train"]
    df_dev = df[df["partition"]=="dev"]
    df_test = df[df["partition"]=="test"]

    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train['word'] = df_train['path'].apply(split_str)
    df_dev['word'] = df_dev['path'].apply(split_str)
    df_test['word'] = df_test['path'].apply(split_str)

    # calculate the nC2 combinations of the words in the dev and test set
    dev_combinations = comb(len(df_dev['word']),2)
    test_combinations = comb(len(df_test['word']),2)
    # print the number of combinations
    print('# combinations in dev set',dev_combinations)
    print('# combinations in test set',test_combinations)

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



    class awe_dataset_pre_computed_new(Dataset):
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
        
    val_data = awe_dataset_pre_computed_new(df_dev)
    test_data = awe_dataset_pre_computed_new(df_test)

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
    print("extracting embedding for val' set")
    val_avg_precision, _, _, _, _ = cal_precision(model, val_loader, device, args.distance)
    print("val average precision:", val_avg_precision)


    print("extracting embedding for test' set")
    test_avg_precision, _, _, _, _ = cal_precision(model, test_loader, device, args.distance)
    print("test average precision:", test_avg_precision)


    # calculate the nC2 combinations of the words in the dev and test set
    dev_combinations = comb(len(df_dev['word']),2)
    test_combinations = comb(len(df_test['word']),2)
    # print the number of combinations
    print("# combinations in dev' set",dev_combinations)
    print("# combinations in test' set",test_combinations)


if __name__ == "__main__":
    main()