"""
Compute the SSL features for a given model and store them

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk
python load_save.py HUBERT_BASE ./data/train.csv ./data/hubert_features/train/ ./data/LibriSpeech/

"""

import argparse
import os
import sys
from utility_functions.feature_extractor import load_model,SSL_features,clip_features
import time
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from os import path
import glob
import librosa



#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#


def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--metadata_file_path", type = str,help = "a text file or dataframe containing paths of wave files, words, start point, duration")
    parser.add_argument('--metadata_file_name', nargs="+", default=["train.csv", "val.csv","test.csv"], help = " list of metadata files")
    parser.add_argument("--path_to_output", type = str, help = "path to output folder where features will be stored")
    parser.add_argument("--path_to_data", type = str, help = "base path to librispeech dataset")

    

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def split_string(strs):
    return strs.split(sep="/")[-2]



#------------------------------#
#      MAIN FUNCTION           #
#------------------------------#

def main():

    args = check_argv()

    print(f"{'metadata_file_path' :<15} : {args.metadata_file_path}")
    print(f"{'metadata_file_name' :<15} : {args.metadata_file_name}")
    print(f"{'path_to_output' :<15} : {args.path_to_output}")
    print(f"{'path_to_data' :<15} : {args.path_to_data}")
    
    # Check whether the specified text file exists or not
    isExist = os.path.exists(args.metadata_file_path)

    if not isExist:
        print(args.metadata_file_path)
        print("provide the correct path for the text/dataframe file having list of wave files")
        sys.exit(1)

    isExist = os.path.exists(args.path_to_data)

    if not isExist:
        print("provide the correct path for the dataset")
        sys.exit(1)

    # Check whether the specified output path exists or not
    isExist = os.path.exists(path.join(args.path_to_output))

    # Create a new directory for output because it does not exist 
    if not isExist:
        os.makedirs(path.join(args.path_to_output))
        print("The new directory for output is created!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.metadata_file_name)
    print("device:", device)


    for f_name in args.metadata_file_name[0].split(','):

        # f_name = args.metadata_file_name[0].split(".")[0]
        f_name = f_name.split(".")[0]
        print(f_name)
        # modify the path to data and path to output
        path_to_data_partition = path.join(args.path_to_data,f_name.split("_")[0],'audio')
        path_to_partition = path.join(args.path_to_output,f_name.split("_")[0])
        # check whether the path to partition exists or not
        isExist = os.path.exists(path_to_partition)
        # Create a new directory for output because it does not exist
        if not isExist:
            os.makedirs(path_to_partition)
            print("The new directory for output is created!")
            print(path_to_partition)
        else:
            print("Feature extraction for the following partition is already done!")
            print(path_to_partition)
            continue


        data = pd.read_csv(os.path.join(args.metadata_file_path,f_name + '.csv'))

        ## Extraction of MFCCs 

        hop_length = 320 # 20 ms shift as sampling rate is 16 Khz for librispeech
        win_length = 480 # 30 ms window length


        print(data.head())
        for _,row in tqdm(data.iterrows()):
            file_path = row['filename'].split('.')[0]
            word_description = row["Label"] + "_" + str(row["Begin"]) + "_" + str(row["End"]) + "_" \
            + file_path
            # add the .file extension
            file_path = file_path + '.flac'
            # take the first name separated by _ as the folder name and append it to the path
            folder_name1, folder_name2 = file_path.split('_')[0], file_path.split('_')[1]
            # append the folder name to the path
            file_path = path.join(path_to_data_partition,folder_name1, folder_name2,file_path)
            # load file

            y, sr = librosa.load(file_path,sr=None)
            magic_number = sr/hop_length
            st_v = int(np.floor(magic_number*row["Begin"]))
            ed_v = int(np.ceil(magic_number*(row['End']-row['Begin'])))
            # compute mfcc

            mfcc = librosa.feature.mfcc(y=y, sr=sr, win_length=win_length, hop_length=hop_length)

            # compute delta

            mfcc_delta = librosa.feature.delta(mfcc)

            # compute delta delta

            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            # clip

            mfcc = mfcc[:, st_v:st_v + ed_v]
            mfcc_delta = mfcc_delta[:, st_v:st_v + ed_v]
            mfcc_delta2 = mfcc_delta2[:, st_v:st_v + ed_v]

            mfcc_dd = np.concatenate((mfcc, mfcc_delta, mfcc_delta2)) # [n_features x seq_len]
            mfcc_dd = mfcc_dd.transpose()                             # [seq_len x n_features]
            word_features  = np.expand_dims(mfcc_dd, axis=0)          # [1, seq_len, n_features]
            word_features = torch.from_numpy(word_features)
            
            # save features
            ## print(word_features.shape)

            torch.save(word_features, path.join(path_to_partition,word_description+".pt"))

    PATH = args.path_to_output
    my_files = sorted(glob.glob(PATH + '*/**/*.pt',recursive=True))
    print("total calculated features files",len(my_files))
    df_metadata = pd.DataFrame(my_files,columns=["path"])
    df_metadata["partition"] = df_metadata["path"].apply(split_string)
    df_metadata.to_csv(os.path.join(PATH,"feature_metadata.csv"),index=False)
    
if __name__ == "__main__":
    main()