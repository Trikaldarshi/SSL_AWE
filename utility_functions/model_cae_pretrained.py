import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import random


# A decoder with attention mechanism:

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)



class DecoderV(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        # self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.fc_out1 = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, (enc_hid_dim * 2) + dec_hid_dim + emb_dim)
        self.fc_out2 = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, (enc_hid_dim * 2) + dec_hid_dim + emb_dim)
        self.fc_out3 = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, 128)
        self.fc_out = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]

        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs, mask)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        # prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        out = self.dropout(F.relu(self.fc_out1(torch.cat((output, weighted, embedded), dim = 1))))
        out = self.dropout(F.relu(self.fc_out2(out)))
        out = self.dropout(F.relu(self.fc_out3(out)))
        prediction = self.fc_out(out)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0), a.squeeze(1)



class model_cae(nn.Module):

  def __init__(self, input_dim, hidden_dim, embedding_dim, rnn_type, bidirectional,
                num_layers, num_classes,dropout, pre_model):
    super(model_cae, self).__init__()

    # # encoder parameters
    # self.input_dim_enc = input_dim
    # self.hidden_dim_enc = hidden_dim
    # self.embedding_dim_enc = embedding_dim

    # # decoder parameters
    # self.input_dim_dec = embedding_dim
    # self.hidden_dim_dec = hidden_dim
    # self.embedding_dim_dec = input_dim
    
    # paratmers for attention block
    self.hidden_dim_enc = hidden_dim

    # decoderV parameters
    self.hidden_dim_decV = hidden_dim
    self.dec_emb_dim = 4*embedding_dim

    assert self.hidden_dim_decV==self.hidden_dim_enc

    # common parameters
    self.rnn_type = rnn_type
    self.bidirectional = bidirectional
    self.num_layers = num_layers
    self.num_classes = num_classes
    self.dropout = dropout
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.encoder = pre_model
    self.scale_encoder = nn.Linear(embedding_dim, hidden_dim)
    self.attention = Attention(self.hidden_dim_enc, self.hidden_dim_decV)
    self.decoderV = DecoderV(self.num_classes, self.dec_emb_dim , self.hidden_dim_enc, self.hidden_dim_decV, self.dropout, self.attention)

  def create_mask(self, src, pad_value = 0):
    mask = (src != pad_value).sum(-1)
    mask = (mask>0).float()
    return mask

  def forward(self, x, lens_x,token, token_len, teacher_forcing_ratio = 0.5):
    encoder_outputs, encoded_x  = self.encoder(x,lens_x)
    encoded_x = torch.tanh(self.scale_encoder(encoded_x))
    # print(encoded_x.shape)

    batch_size = token.size(0) # [batch_size, token_len]
    trg = token.T # [token len, batch size]
    trg_len = trg.size(0) # [token_len]
    # print("trg_len:", trg_len)

    hidden = encoded_x

    outputs = torch.zeros(trg_len, batch_size, self.num_classes).to(self.device)

    # input = torch.zeros(batch_size, dtype=torch.int32).to(self.device) # [batch_size]
    input = trg[0,:]
    mask = self.create_mask(x)

    for t in range(1, trg_len):
    
        #insert input token embedding, previous hidden state and all encoder hidden states
        #receive output tensor (predictions) and new hidden state
        output, hidden,_ = self.decoderV(input, hidden, encoder_outputs, mask)
        
        #place predictions in a tensor holding predictions for each token
        outputs[t] = output
        
        #decide if we are going to use teacher forcing or not
        teacher_force = random.random() < teacher_forcing_ratio
        # print(teacher_force)        
        #get the highest predicted token from our predictions
        top1 = output.argmax(1) 
        
        #if teacher forcing, use actual next token as next input
        #if not, use predicted token
        # print(top1.shape)
        # print(trg[t].shape)
        input = trg[t] if teacher_force else top1.detach()


    return outputs