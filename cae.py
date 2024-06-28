"""
train CAE-RNN network and save the best model

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

"""

import argparse
import os
import random
import sys
import time
from distutils.util import strtobool
from os import path
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd

from utility_functions.awe_dataset_class import (awe_dataset_pre_computed,
                                                 cae_awe_dataset_pre_computed)
from utility_functions.model_cae import model_cae
from utility_functions.utils_function import (average_precision, collate_fn,
                                              collate_fn_cae)

possible_models = ["HUBERT_BASE","HUBERT_LARGE","HUBERT_XLARGE","WAV2VEC2_BASE","WAV2VEC2_LARGE",
                    "WAV2VEC2_LARGE_LV60K","WAV2VEC2_XLSR53","HUBERT_ASR_LARGE","HUBERT_ASR_XLARGE",
                    "WAV2VEC2_ASR_BASE_10M","WAV2VEC2_ASR_BASE_100H","WAV2VEC2_ASR_BASE_960H",
                    "WAV2VEC2_ASR_LARGE_10M","WAV2VEC2_ASR_LARGE_100H","WAV2VEC2_ASR_LARGE_960H",
                    "WAV2VEC2_ASR_LARGE_LV60K_10M","WAV2VEC2_ASR_LARGE_LV60K_100H","WAV2VEC2_ASR_LARGE_LV60K_960H",
                    "MFCC",'WAVLM_BASE','WAVLM_LARGE', 'WAV2VEC2_XLSR_300M']


#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#


def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--model_name", type=str, help = "name of the model for example, HUBERT_BASE", 
                        nargs='?', default = "HUBERT_BASE", choices = possible_models)
    parser.add_argument("--input_dim", type = int, help = "dimension of input features", nargs='?', 
                        default=768)
    parser.add_argument("--metadata_file", type = str, help = "a text file or dataframe containing paths  \
      of wave files, words, start point, duration or SSL features metadata file")
    parser.add_argument("--path_to_output", type = str, help = "path to output folder where features will be stored", 
                        nargs='?',default = "./output")
    parser.add_argument("--layer", type = int, help = "layer you want to extract, type mfcc for mfcc", 
                        nargs='?',default=0)
    parser.add_argument("--lr", type = float, help = "learning rate", nargs='?', default=0.001)
    parser.add_argument("--batch_size", type = int, help = "batch_size", nargs='?', default=2)
    parser.add_argument("--n_epochs", type = int, help = "number of epochs", nargs='?', default=10)

    parser.add_argument("--pre_compute", type=lambda x:bool(strtobool(x)), nargs='?', const=True, 
                        default=True, help = "use pre computed features or not")

    parser.add_argument("--step_lr", type = int, help = "steps at which learning rate will decrease",
                        nargs='?',default = 20)
    parser.add_argument("--embedding_dim", type = int, help = "value of embedding dimensions",nargs='?',
                        default = 128)
    parser.add_argument("--distance", type = str, help = "type of distance to compute the similarity",nargs='?',
                        default = "cosine")
    parser.add_argument("--opt", type = str, help = "optimizer", nargs='?', default = "adam", 
                        choices=["adam","sgd"])
    parser.add_argument("--loss", type = str, help = "loss function", nargs='?', default = "mse", 
                        choices=["mse","mae"])
    parser.add_argument("--hidden_dim", type = int, help = "rnn hidden dimension values", default=512)
    parser.add_argument("--rnn_type", type = str, help = " type or rnn, gru or lstm?", default="LSTM", 
                        choices=["GRU","LSTM"])
    parser.add_argument("--bidirectional", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False, 
                        help = " bidirectional rnn or not")
    parser.add_argument("--num_layers", type = int, help = " number of layers in rnn network, input more than 1", 
                        default=2) 
    parser.add_argument("--dropout", type = float, help = "dropout applied inside rnn network", default=0.2)
    parser.add_argument("--wandb", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False, 
                        help = "use wandb to log your progress or not")
    parser.add_argument("--checkpoint_model", type = str, help = "path to model checkpoints", nargs='?',
                        default = "/saved_model")
    

    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])


def cal_precision(model,loader,device,distance):
  embeddings, words = [], []
  model = model.eval()
  with torch.no_grad():
    for _, (data,lens,word_name,_) in enumerate(loader):

      lens, perm_idx = lens.sort(0, descending=True)
      data = data[perm_idx]
      word_name = word_name[perm_idx]
      
      data, lens  = data.to(device), lens.to(device)
      # lens = lens.to(device)

      _ ,emb = model.encoder(data, lens)
      embeddings.append(emb)
      words.append(word_name)
  words = np.concatenate(words)
  uwords = np.unique(words)
  word2id = {v: k for k, v in enumerate(uwords)}
  ids = [word2id[w] for w in words]
  embeddings, ids = torch.cat(embeddings,0).to(torch.float16), np.array(ids)
  avg_precision,_ = average_precision(embeddings.cpu(),ids, distance)

  return avg_precision


#------------------------------#
#      MAIN FUNCTION           #
#------------------------------#

def main():
  
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


  ## read the parsed arguments

  args = check_argv()

  print(f"{'model_name' :<20} : {args.model_name}")
  print(f"{'input_dim' :<20} : {args.input_dim}")
  print(f"{'metadata_file' :<20} : {args.metadata_file}")
  print(f"{'path_to_output' :<20} : {args.path_to_output}")
  print(f"{'layer' :<20} : {args.layer}")
  print(f"{'pre_compute' :<20} : {args.pre_compute}")
  print(f"{'lr' :<20} : {args.lr}")
  print(f"{'batch_size' :<20} : {args.batch_size}")
  print(f"{'n_epochs' :<20} : {args.n_epochs}")
  print(f"{'step_lr' :<20} : {args.step_lr}")
  print(f"{'embedding_dim' :<20} : {args.embedding_dim}")
  print(f"{'distance' :<20} : {args.distance}")
  print(f"{'opt' :<20} : {args.opt}")
  print(f"{'loss' :<20} : {args.loss}")
  print(f"{'hidden_dim' :<20} : {args.hidden_dim}")
  print(f"{'rnn_type' :<20} : {args.rnn_type}")
  print(f"{'bidirectional' :<20} : {args.bidirectional}")
  print(f"{'num_layers' :<20} : {args.num_layers}")
  print(f"{'dropout' :<20} : {args.dropout}")
  print(f"{'wandb' :<20} : {args.wandb}")
  print(f"{'checkpoint_model' :<20} : {args.checkpoint_model}")


  # Check whether the specified text/dataframe metadata file exists or not
  isExist = os.path.exists(args.metadata_file)

  if not isExist:
      print(args.metadata_file)
      print("provide the correct path for the text/dataframe file having list of wave files")
      sys.exit(1)

  # copy the metadata file to the scratch directory, ignore this

  ###############################################
  # temp_dir = tempfile.gettempdir()
  # print('scratch dir',temp_dir)
  # sub_dir = args.metadata_file.split('/')[7]
  # os.system('mkdir -p' + ' ' + os.path.join(temp_dir,sub_dir))
  # print('created sub_dir',os.path.join(temp_dir,sub_dir))
  # print('copying files from', ('/').join(args.metadata_file.split('/')[:-1]), 'to', os.path.join(temp_dir,sub_dir))

  # os.system('cp -r' + ' ' + ('/').join(args.metadata_file.split('/')[:-1]) + ' ' + os.path.join(temp_dir,sub_dir))

  # args.metadata_file = '/'.join(args.metadata_file.split('/')[7:])
  # args.metadata_file = os.path.join(temp_dir, args.metadata_file)

  # df = pd.read_csv(args.metadata_file)
  # print(df.head())
  # print(df.shape)

  # def mod_path(path,temp_dir):
  #     path = path.split('/')
  #     path = path[7:]
  #     path = '/'.join(path)
  #     path = os.path.join(temp_dir,path)
  #     return path

  # df['path'] = df['path'].apply(lambda x: mod_path(x,temp_dir))
  # print(df.head())
  # print(df.shape)

  # df.to_csv(args.metadata_file,index=False)
  # print('saved to',args.metadata_file)

  #############################################



  # Check whether the specified output path exists or not
  isExist = os.path.exists(args.path_to_output)
  
  # Create a new directory for output if it does not exist 
  if not isExist:
      os.makedirs(args.path_to_output)
      print("The new directory for output is created!")

  # make sure the output directory is ALREADY created BEFORE initializing wandb
  if args.wandb:
    wandb.init(project="asru-awe-2023-final-woc", resume=False, dir=args.path_to_output)
    wandb.config.update(args)

  ## create a unique output checkpoint storage location with argument files
  args.path_to_output = path.join(args.path_to_output,args.checkpoint_model)
  isExist = os.path.exists(args.path_to_output)

  if not isExist:
    os.makedirs(args.path_to_output)
    print("The new directory for saving checkpoint is created!")

    with open(path.join(args.path_to_output,'config.txt'), 'w') as f:
      for key, value in vars(args).items(): 
              f.write('--%s=%s\n' % (key, value))
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  batch_size = args.batch_size

  # print("available device:",device)
  print("Is device CUDA:", device.type=="cuda")
  if device.type == "cuda":
      num_workers = 8
      pin_memory = True
  else:
      num_workers = 0
      pin_memory = False

  print("number of workers:", num_workers)
  print("pin memory status:", pin_memory)
  if args.pre_compute:

    print("using pre-computed features", args.model_name)
    
    train_data = cae_awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="train"
    )
    val_data = awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="dev"
    )
    test_data = awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="test"
    )
  else:
    print("not compatible with on the fly computation")
    sys.exit(1)

  # print("length of training data:",len(train_data))
  # print("length of validation data:",len(val_data))
  # print("length of test data:",len(test_data))
  # For debugging purpose: use a subset of the data  

  # indices = np.random.choice(range(len(train_data)), 1000, replace=False)
  # train_data = torch.utils.data.Subset(train_data, indices)
  # indices = np.random.choice(range(len(val_data)), 1000, replace=False)
  # val_data = torch.utils.data.Subset(val_data, indices)
  # test_data = torch.utils.data.Subset(test_data, indices)

  print("length of training data:",len(train_data))
  print("length of validation data:",len(val_data))
  print("length of test data:",len(test_data))
  
  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=collate_fn_cae,
      drop_last = False,
      num_workers=num_workers,
      pin_memory=pin_memory,
      worker_init_fn=seed_worker,
      generator=g
  )
  val_loader = torch.utils.data.DataLoader(
      val_data,
      batch_size=batch_size,
      shuffle=True,
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
      shuffle=True,
      collate_fn=collate_fn,
      drop_last = False,
      num_workers=num_workers,
      pin_memory=pin_memory,
      worker_init_fn=seed_worker,
      generator=g
  )

  ## Define the model

  model = model_cae(args.input_dim, args.hidden_dim, args.embedding_dim, args.rnn_type, args.bidirectional, 
                    args.num_layers, args.dropout)
  model = model.to(device)
  model_description = '_'.join([args.model_name, str(args.embedding_dim)])

  # Check whether the specified checkpoint/pre-trained model exists or not
  PATH = path.join(args.path_to_output, model_description + ".pt")
  PATH_BEST = path.join(args.path_to_output, model_description + "_BEST.pt")
  isCheckpoint = os.path.exists(PATH)

  if isCheckpoint:
    print("loading the saved checkpoint for training:")
  else:
    print("no checkpoint given, starting training from scratch!")


  def train_model(model, train_load, val_load, n_epochs):

    if args.opt=="sgd":
      optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.5)
    if args.loss=="mse":
      criterion = nn.MSELoss().to(device)
    else:
      criterion = nn.L1Loss().to(device)

    if isCheckpoint==True:
      print("recent checkpoint:")
      checkpoint = torch.load(PATH,map_location=torch.device(device))
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict']),
      base_epoch = checkpoint['epoch']
      history = checkpoint['loss_history']
      best_ap = history['best_ap']
      best_epoch = history['best_epoch']
      optimizer.param_groups[0]['capturable'] = True # Error: assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors.
      scheduler.step()
      base_epoch += 1


    else:
      history = dict(train_loss=[], val_loss=[], val_avg_precision=[],best_epoch=0,best_ap=0)
      base_epoch = 1
      best_ap = 0.0
      best_epoch = 0
    print("training starting at epoch - ", base_epoch)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(base_epoch, n_epochs + 1):
      model = model.train()

      train_losses = []
      start_epoch = time.time()
      for ij, (x, lens_x, _,_, y, lens_y,_,_) in enumerate(train_load): ## CAE training

        optimizer.zero_grad()
        x, lens_x, y, lens_y = x.to(device), lens_x.to(device), y.to(device), lens_y.to(device)
        lens_x, perm_idx = lens_x.sort(0, descending=True)
        x = x[perm_idx]
        y = y[perm_idx]
        lens_y = lens_y[perm_idx]

        seq_pred = model(x, lens_x, lens_y)
        loss = criterion(seq_pred, y)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

      print("Time for one epoch:",epoch, time.time()-start_epoch)


      val_losses = []
      model = model.eval()
      with torch.no_grad():
        for _, (data,lens,_,_) in enumerate(val_load):
          data, lens = data.to(device), lens.to(device)
          lens, perm_idx = lens.sort(0, descending=True)
          data = data[perm_idx]

          seq_pred = model(data, lens, lens)

          loss = criterion(seq_pred, data) # data[0] will give s1-[20,768]; s2-[2,768]; s3-[3,768]:--------> [25,768]
                                                      # data_packed.data will return the [[25,768]] thing only
          val_losses.append(loss.item())
        

      val_avg_precision = cal_precision(model, val_load, device, args.distance)
      train_loss = np.mean(train_losses)
      val_loss = np.mean(val_losses)

      history['train_loss'].append(train_loss)
      history['val_loss'].append(val_loss)
      history["val_avg_precision"].append(val_avg_precision)

      if val_avg_precision > best_ap:
        best_ap = val_avg_precision
        best_epoch = epoch
        history["best_epoch"] = best_epoch
        history["best_ap"] = best_ap
        print("checkpoint saved for best epoch for average precision")
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': history
        }, PATH_BEST)
        
      if epoch % 1 == 0:
        print("checkpoint logging :")
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': history
        }, PATH)


      print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss} ; best epoch {best_epoch} val avg precision {val_avg_precision}')
      print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

      if args.wandb:
        wandb.log({"train loss": train_loss, "val loss":val_loss, 
                    "val avg precision": val_avg_precision, "lr_history":optimizer.param_groups[0]['lr'], "best epoch":best_epoch})
      scheduler.step()


  train_model(
    model,
    train_loader, 
    val_loader, 
    n_epochs=args.n_epochs
  )

  # Load the best model
  checkpoint_best = torch.load(PATH_BEST,map_location=torch.device(device))
  model.load_state_dict(checkpoint_best['model_state_dict'])
  history = checkpoint_best['loss_history']
  best_epoch = history["best_epoch"]
  best_ap = history["best_ap"]
  print("best average precision on val set:", best_ap)
  model.eval()
  test_avg_precision = cal_precision(model, test_loader, device, args.distance)

  print("average precision on test set:", test_avg_precision)
  print("average precision on val set:", best_ap)
  print("best epoch:", best_epoch)
  print(" We are done! Bye Bye. Have a nice day!")

  if args.wandb:
    wandb.log({"test ap": test_avg_precision})
    wandb.log({"val ap": best_ap})

if __name__ == "__main__":
    main()

