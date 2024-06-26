"""
Compute the SSL features for a given model and store them

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

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
import torchaudio

possible_models = ["HUBERT_BASE","HUBERT_LARGE","HUBERT_XLARGE","WAV2VEC2_BASE","WAV2VEC2_LARGE",
                    "WAV2VEC2_LARGE_LV60K","WAV2VEC2_XLSR53","HUBERT_ASR_LARGE","HUBERT_ASR_XLARGE",
                    "WAV2VEC2_ASR_BASE_10M","WAV2VEC2_ASR_BASE_100H","WAV2VEC2_ASR_BASE_960H",
                    "WAV2VEC2_ASR_LARGE_10M","WAV2VEC2_ASR_LARGE_100H","WAV2VEC2_ASR_LARGE_960H",
                    "WAV2VEC2_ASR_LARGE_LV60K_10M","WAV2VEC2_ASR_LARGE_LV60K_100H","WAV2VEC2_ASR_LARGE_LV60K_960H",
                    'WAV2VEC2_XLSR_300M','WAVLM_BASE','WAVLM_LARGE']


#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#


def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--model",type=str,help = "name of the model for example, HUBERT_BASE",choices = possible_models)
    parser.add_argument("--metadata_file_path", type = str,help = "a text file or dataframe containing paths of wave files, words, start point, duration")
    parser.add_argument('--metadata_file_name', nargs="+", default=["train.csv", "val.csv","test.csv"], help = " list of metadata files")
    parser.add_argument("--path_to_output", type = str, help = "path to output folder where features will be stored")
    parser.add_argument("--path_to_data", type = str, help = "base path to librispeech dataset")
    parser.add_argument("--layer", type = int, help = "layer you want to extract",nargs='?',default=12)
    

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

    print(f"{'model' :<15} : {args.model}")
    print(f"{'metadata_file_path' :<15} : {args.metadata_file_path}")
    print(f"{'metadata_file_name' :<15} : {args.metadata_file_name}")
    print(f"{'path_to_output' :<15} : {args.path_to_output}")
    print(f"{'path_to_data' :<15} : {args.path_to_data}")
    print(f"{'layer' :<15} : {args.layer}")
    
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
    model,sr = load_model(args.model,device)
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
            # check if the file exists
            isExist = os.path.exists(file_path)
            if not isExist:
                print("The file does not exist!")
                sys.exit(1)

            ###############
            # Load the waveform
            waveform, sample_rate = torchaudio.load(file_path)
            # Resample the waveform
            waveform = torchaudio.functional.resample(waveform, sample_rate, sr).to(device)
            waveform = waveform[:,int(np.floor(row["Begin"]*sr)):int(np.ceil(row["End"]*sr))]
            # Extract acoustic features
            features, _ = model.extract_features(waveform)
            word_features = features[args.layer-1].detach().cpu()
            ###############
            # features = SSL_features(file_path,model,sr,layer=args.layer,device=device)
            # word_features = clip_features(features,row["Begin"],row['End']-row['Begin'],layer=args.layer).detach().cpu()
            torch.save(word_features, path.join(path_to_partition,word_description + ".pt"))


    PATH = args.path_to_output
    my_files = sorted(glob.glob(PATH + '*/**/*.pt',recursive=True))
    print("total calculated features files",len(my_files))
    df_metadata = pd.DataFrame(my_files,columns=["path"])
    df_metadata["partition"] = df_metadata["path"].apply(split_string)
    df_metadata.to_csv(os.path.join(PATH,"feature_metadata.csv"),index=False)

if __name__ == "__main__":
    main()