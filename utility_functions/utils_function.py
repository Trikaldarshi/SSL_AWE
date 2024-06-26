#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#
from os import path

import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.special import comb


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    # batch = [item for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, lengths, word_names, sp_ch_ut_ids = [], [], [], []

    # Gather in lists, and encode labels as indices
    for hubert_features,lens,wn,identifier in batch:
        tensors += [hubert_features]
        lengths += [lens]
        word_names += [wn]
        sp_ch_ut_ids += [identifier]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    lengths = torch.stack(lengths)
    word_names = np.array(word_names)
    sp_ch_ut_ids = np.array(sp_ch_ut_ids)

    return tensors, lengths, word_names, sp_ch_ut_ids

def collate_fn_cae(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors_x, lengths_x, word_names_x, sp_ch_ut_ids_x = [], [], [], []
    tensors_y, lengths_y, word_names_y, sp_ch_ut_ids_y = [], [], [], []

    # Gather in lists, and encode labels as indices
    for hubert_features_x,lens_x,wn_x,identifier_x, hubert_features_y,lens_y,wn_y,identifier_y in batch:
        tensors_x += [hubert_features_x]
        lengths_x += [lens_x]
        word_names_x += [wn_x]
        sp_ch_ut_ids_x += [identifier_x]

        tensors_y += [hubert_features_y]
        lengths_y += [lens_y]
        word_names_y += [wn_y]
        sp_ch_ut_ids_y += [identifier_y]


    # Group the list of tensors into a batched tensor
    tensors_x = pad_sequence(tensors_x)
    lengths_x = torch.stack(lengths_x)
    word_names_x = np.array(word_names_x)
    sp_ch_ut_ids_x = np.array(sp_ch_ut_ids_x)

    # Group the list of tensors into a batched tensor
    tensors_y = pad_sequence(tensors_y)
    lengths_y = torch.stack(lengths_y)
    word_names_y = np.array(word_names_y)
    sp_ch_ut_ids_y = np.array(sp_ch_ut_ids_y)

    return tensors_x, lengths_x, word_names_x, sp_ch_ut_ids_x, tensors_y, lengths_y, word_names_y, sp_ch_ut_ids_y

def save_checkpoints(epoch,model,optimizer,scheduler,history,PATH):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss_history': history
    }, PATH)
    return print("checkpoint saved at epoch:",epoch)

def load_checkpoints(model,optimizer,scheduler,PATH,device):
    checkpoint = torch.load(PATH,map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict']),
    epoch = checkpoint['epoch']
    history = checkpoint['loss_history']
    print("checkpoint loaded at epoch:",epoch)
    return model, optimizer, scheduler, epoch, history

def average_precision(data, labels, metric = "cosine", show_plot=False):
    """
    Calculate average precision and precision-recall breakeven, and return
    the average precision / precision-recall breakeven calculated
    using `same_dists` and `diff_dists`.
    -------------------------------------------------------------------
    returns average_precision, precision-recall break even : (float, float)
    """
    num_examples = len(labels)
    num_pairs = int(comb(num_examples, 2))


    # build up binary array of matching examples
    matches = np.zeros(num_pairs, dtype=bool)

    i = 0
    for n in range(num_examples):
        j = i + num_examples - n - 1
        matches[i:j] = (labels[n] == labels[n + 1:]).astype(np.int32)
        i = j

    num_same = np.sum(matches)

    # calculate pairwise distances and sort matches
    dists = pdist(data, metric=metric)
    matches = matches[np.argsort(dists)]


    # calculate precision, average precision, and recall
    precision = np.cumsum(matches) / np.arange(1, num_pairs + 1)
    average_precision = np.sum(precision * matches) / num_same
    recall = np.cumsum(matches) / num_same

    # multiple precisions can be at single recall point, take max
    for n in range(num_pairs - 2, -1, -1):
        precision[n] = max(precision[n], precision[n + 1])

    # calculate precision-recall breakeven
    prb_ix = np.argmin(np.abs(recall - precision))
    prb = (recall[prb_ix] + precision[prb_ix]) / 2.
    if show_plot:
        import matplotlib.pyplot as plt
        print("plot created")
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig('foo.pdf')
        

    return average_precision, prb
    

def metric2(data1, data2, labels, metric = "cosine"):
    """
    Calculate average precision and precision-recall breakeven, and return
    the average precision / precision-recall breakeven calculated
    using `same_dists` and `diff_dists`.
    -------------------------------------------------------------------
    returns average_precision, precision-recall break even : (float, float)
    """
    num_examples = len(labels)
    num_same = np.sum(labels)

    # calculate pairwise distances and sort matches
    dists = cdist(data1, data2, metric=metric).flatten()
    labels = labels[np.argsort(dists)]

    # calculate precision, average precision, and recall
    precision = np.cumsum(labels) / np.arange(1, num_examples + 1)
    average_precision = np.sum(precision * labels) / num_same
    recall = np.cumsum(labels) / num_same

    # multiple precisions can be at single recall point, take max
    for n in range(num_examples - 2, -1, -1):
        precision[n] = max(precision[n], precision[n + 1])

    # calculate precision-recall breakeven
    prb_ix = np.argmin(np.abs(recall - precision))
    prb = (recall[prb_ix] + precision[prb_ix]) / 2.

    return average_precision, prb

def metric3(data1, data2, labels, metric = "cosine"):
    """
    Calculate average precision and precision-recall breakeven, and return
    the average precision / precision-recall breakeven calculated
    using `same_dists` and `diff_dists`.
    -------------------------------------------------------------------
    returns average_precision, precision-recall break even : (float, float)
    """
    num_examples = len(labels)
    num_same = np.sum(labels)

    # calculate pairwise distances and sort matches
    dists = cdist(data1, data2, metric=metric)
    labels = labels[np.argsort(dists)]

    # calculate precision, average precision, and recall
    precision = np.cumsum(labels) / np.arange(1, num_examples + 1)
    average_precision = np.sum(precision * labels) / num_same
    recall = np.cumsum(labels) / num_same

    # multiple precisions can be at single recall point, take max
    for n in range(num_examples - 2, -1, -1):
        precision[n] = max(precision[n], precision[n + 1])

    # calculate precision-recall breakeven
    prb_ix = np.argmin(np.abs(recall - precision))
    prb = (recall[prb_ix] + precision[prb_ix]) / 2.

    return average_precision, prb


def collate_fn_pre_training(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, lengths, word_names, sp_ch_ut_ids, token, token_len = [], [], [], [], [], []

    # Gather in lists, and encode labels as indices
    for hubert_features,lens,wn,identifier,t,t_l in batch:
        tensors += [hubert_features]
        lengths += [lens]
        word_names += [wn]
        sp_ch_ut_ids += [identifier]
        token += [t]
        token_len += [t_l] 

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    lengths = torch.stack(lengths)
    word_names = np.array(word_names)
    sp_ch_ut_ids = np.array(sp_ch_ut_ids)
    token = np.array(token,dtype=object)
    token_len = np.array(token_len)

    return tensors, lengths, word_names, sp_ch_ut_ids, token, token_len

def collate_fn_cae_pre_training(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors_x, lengths_x, word_names_x, sp_ch_ut_ids_x, token_x, token_len = [], [], [], [], [], []
    tensors_y, lengths_y, word_names_y, sp_ch_ut_ids_y, token_y = [], [], [], [], []

    # Gather in lists, and encode labels as indices
    for hubert_features_x,lens_x,wn_x,identifier_x, t_x, t_l, hubert_features_y,lens_y,wn_y,identifier_y, t_y in batch:
        tensors_x += [hubert_features_x]
        lengths_x += [lens_x]
        word_names_x += [wn_x]
        sp_ch_ut_ids_x += [identifier_x]
        token_x +=[t_x]
        token_len +=[t_l]

        tensors_y += [hubert_features_y]
        lengths_y += [lens_y]
        word_names_y += [wn_y]
        sp_ch_ut_ids_y += [identifier_y]
        token_y +=[t_y]


    # Group the list of tensors into a batched tensor
    tensors_x = pad_sequence(tensors_x)
    lengths_x = torch.stack(lengths_x)
    word_names_x = np.array(word_names_x)
    sp_ch_ut_ids_x = np.array(sp_ch_ut_ids_x)
    token_x = np.array(token_x,dtype=object)
    token_len = np.array(token_len)

    # Group the list of tensors into a batched tensor
    tensors_y = pad_sequence(tensors_y)
    lengths_y = torch.stack(lengths_y)
    word_names_y = np.array(word_names_y)
    sp_ch_ut_ids_y = np.array(sp_ch_ut_ids_y)
    token_y = np.array(token_y,dtype=object)

    return tensors_x, lengths_x, word_names_x, sp_ch_ut_ids_x, token_x, token_len, \
           tensors_y, lengths_y, word_names_y, sp_ch_ut_ids_y, token_y
