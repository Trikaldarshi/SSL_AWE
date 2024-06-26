import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

### Simplest multi-layer bidirectional CAE-RNN
class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, embedding_dim, rnn_type, bidirectional,
                num_layers, dropout):
    super(Encoder, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.embedding_dim = embedding_dim
    self.rnn_type = rnn_type
    self.bidirectional = bidirectional
    self.dropout = dropout
    self.num_layers = num_layers
    if self.rnn_type=="LSTM":
      self.rnn_enc = nn.LSTM(
        input_size=self.input_dim,
        hidden_size=self.hidden_dim,
        num_layers=self.num_layers,
        batch_first=True,
        bidirectional=self.bidirectional,
        dropout=self.dropout
      )
    else:
      self.rnn_enc = nn.GRU(
        input_size=self.input_dim,
        hidden_size=self.hidden_dim,
        num_layers=self.num_layers,
        batch_first=True,
        bidirectional=self.bidirectional,
        dropout=self.dropout
      )   
    if self.bidirectional: 
      self.num_directions = 2
    else:
      self.num_directions = 1
      
    self.fc_enc = nn.Linear(self.hidden_dim*self.num_directions, self.embedding_dim)

    self.tanh = nn.Tanh()

  def forward(self, x, lens):

    x =  torch.nn.utils.rnn.pack_padded_sequence(x, lens.to('cpu'), batch_first=True) # [batch,length,features]
    if self.rnn_type=="LSTM":
      _, (h_enc, _) = self.rnn_enc(x)
    else:
      o_enc, h_enc = self.rnn_enc(x)

    # Unpack the output sequence
    outputs, _ = pad_packed_sequence(o_enc, batch_first=False) 

    # concatenate the last forward and backward hidden state
    if self.bidirectional:
      h_enc_concat = torch.cat((h_enc[-2,:,:], h_enc[-1,:,:]), dim = 1)
    else:
      h_enc_concat = h_enc[-1].squeeze()

    # apply transformation

    emb_enc = self.fc_enc(h_enc_concat)

    # apply tanh to map the embeddings between -1 to +1 
    emb_enc = self.tanh(emb_enc)

    return outputs, emb_enc



class Decoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, embedding_dim, rnn_type, bidirectional,
                num_layers, dropout, device):
    super(Decoder, self).__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.embedding_dim = embedding_dim
    self.rnn_type = rnn_type
    self.bidirectional = bidirectional
    self.num_layers = num_layers
    self.dropout = dropout
    self.device = device

    if self.rnn_type == "LSTM":
      self.rnn_dec = nn.LSTM(
        input_size=self.input_dim,
        hidden_size=self.hidden_dim,
        num_layers=self.num_layers,
        batch_first=True,
        bidirectional=self.bidirectional,
        dropout=self.dropout
      )
    else:
      self.rnn_dec = nn.GRU(
        input_size=self.input_dim,
        hidden_size=self.hidden_dim,
        num_layers=self.num_layers,
        batch_first=True,
        bidirectional=self.bidirectional,
        dropout=self.dropout
      )
    if self.bidirectional: 
      self.num_directions = 2
    else:
      self.num_directions = 1
    self.fc_dec = nn.Linear(self.hidden_dim*self.num_directions, self.embedding_dim)

    
  def forward(self, x, lens):
    x = torch.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)  # lens is a list here so no need of .to("cpu")
    if self.rnn_type=="LSTM":
      out_rnn_dec, (_, _) = self.rnn_dec(x)
    else:
      out_rnn_dec, _ = self.rnn_dec(x)

    
    # to input the unpacked output sequence with zero
    seq_unpacked_dec, lens = pad_packed_sequence(out_rnn_dec, batch_first=True)
    final_output = self.fc_dec(seq_unpacked_dec)

    # # Create mask that replace values in rows wirh zeros at indices greater that original seq_len
    # # mask = [torch.ones(lens[i],final_output.size(-1)) for i in range(final_output.size(0))]
    # # mask_padded = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(self.device)

    mask = torch.repeat_interleave(torch.ones(final_output.size(0),final_output.size(-1)), lens, dim=0, output_size=sum(lens))
    mask = torch.split(mask, lens.tolist())
    mask_padded = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(self.device)

    # Apply mask to final_output
    final_output_masked = torch.mul(final_output, mask_padded)

    return final_output_masked
    # return final_output



class model_cae(nn.Module):

  def __init__(self, input_dim, hidden_dim, embedding_dim, rnn_type, bidirectional,
                num_layers, dropout):
    super(model_cae, self).__init__()

    # encoder parameters
    self.input_dim_enc = input_dim
    self.hidden_dim_enc = hidden_dim
    self.embedding_dim_enc = embedding_dim

    # decoder parameters
    self.input_dim_dec = embedding_dim
    self.hidden_dim_dec = hidden_dim
    self.embedding_dim_dec = input_dim

    # common parameters
    self.rnn_type = rnn_type
    self.bidirectional = bidirectional
    self.num_layers = num_layers
    self.dropout = dropout
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.encoder = Encoder(self.input_dim_enc, self.hidden_dim_enc, 
                                  self.embedding_dim_enc, self.rnn_type, 
                                  self.bidirectional, self.num_layers, self.dropout)
    self.decoder = Decoder(self.input_dim_dec, self.hidden_dim_dec,
                                  self.embedding_dim_dec, self.rnn_type,
                                  self.bidirectional, self.num_layers, self.dropout, self.device)

  def forward(self, x, lens_x, lens_y):
    encoder_outputs, encoded_x  = self.encoder(x,lens_x)

    # Decoding
    # repeat latent embedding as input to the rnn up to corresponding output sequence length (corresponding lengths are not sorted)

    lengths_sorted = [(length,idx)  for (idx,length) in sorted(enumerate(lens_y), key=lambda x:x[1], reverse=True)]
    corr_lengths_sorted = [x[0].tolist() for x in lengths_sorted]
    corr_sorting_indices = [x[1] for x in lengths_sorted] # use to rearange lens_y to orignal sequence
    encoded_x = encoded_x[corr_sorting_indices]

    # Prepare and pad the correspondence sequences

    corr_sequences = torch.repeat_interleave(encoded_x, torch.tensor(corr_lengths_sorted).to(self.device), dim=0, output_size=sum(corr_lengths_sorted))
    corr_sequences = torch.split(corr_sequences, corr_lengths_sorted)
    corr_sequences_padded = torch.nn.utils.rnn.pad_sequence(corr_sequences, batch_first=True) # pad variable length sequences to max seq length

    # Decode padded sequences
    decoded_x = self.decoder(corr_sequences_padded, corr_lengths_sorted)

    # Reorder output in orignal order ---- just debug once to make it faster
    final_decoded_x = torch.zeros_like(decoded_x)
    for i in range(len(decoded_x)):                
        final_decoded_x[corr_sorting_indices[i]] = decoded_x[i]

    return final_decoded_x