import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval

class dataset_embedding(Dataset):
    def __init__(self, metadata, embedding_mat):
        
        self.check = torch.cuda.is_available()
        self.metadata = pd.read_csv(metadata, converters={'tokenized': literal_eval})
        if self.check:
            self.embedding = torch.load(embedding_mat)
        else:
            self.embedding  = torch.load(embedding_mat,map_location=torch.device('cpu'))
    
        

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        unique_id = self.metadata.iloc[idx, 0]
        word = self.metadata.iloc[idx, 1]
        id = self.metadata.iloc[idx, 2]
        tokens = self.metadata.iloc[idx, 3]
        emb = self.embedding[idx]

            

        return emb, tokens, id, word, unique_id

class fact_net(nn.Module):

    def __init__(self, input_dim, sub_emb, num_classes):
        super(fact_net, self).__init__()
        
        self.input_dim = input_dim
        self.sub_emb = sub_emb
        self.num_classes = num_classes
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.input_dim, self.sub_emb) 
        self.fc11 = nn.Linear(self.sub_emb, self.num_classes)
        self.fc2 = nn.Linear(self.input_dim, self.sub_emb)
        self.fc22 = nn.Linear(self.sub_emb, self.num_classes)
        self.fc3 = nn.Linear(self.input_dim, self.sub_emb)
        self.fc33 = nn.Linear(self.sub_emb, self.num_classes)
        self.fc_projection = nn.Linear(self.sub_emb, self.input_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        
        x1 = self.dropout(torch.tanh(self.fc1(x)))
        x11 = self.fc11(x1)
        x2 = self.dropout(torch.tanh(self.fc2(x)))
        x22 = self.fc22(x2)
        x3 = self.dropout(torch.tanh(self.fc3(x)))
        x33 = self.fc33(x3)
        sum_emb = x1+x2+x3
        x_recon = torch.tanh(self.fc_projection(sum_emb))
        return x1, x2, x3, x11, x22, x33, x_recon

### 3 feed forward layers with no reconstruction loss

class fact_net2(nn.Module):

    def __init__(self, input_dim, sub_emb, num_classes):
        super(fact_net2, self).__init__()
        
        self.input_dim = input_dim
        self.sub_emb = sub_emb
        self.num_classes = num_classes
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.input_dim, 4*self.input_dim) 
        self.fc11 = nn.Linear(4*self.input_dim, 4*self.input_dim)
        self.fc111 = nn.Linear(4*self.input_dim, self.sub_emb)
        self.fc1111 = nn.Linear(self.sub_emb, self.num_classes)

        self.fc2 = nn.Linear(self.input_dim, 4*self.input_dim) 
        self.fc22 = nn.Linear(4*self.input_dim, 4*self.input_dim)
        self.fc222 = nn.Linear(4*self.input_dim, self.sub_emb)
        self.fc2222 = nn.Linear(self.sub_emb, self.num_classes)

        self.fc3 = nn.Linear(self.input_dim, 4*self.input_dim) 
        self.fc33 = nn.Linear(4*self.input_dim, 4*self.input_dim)
        self.fc333 = nn.Linear(4*self.input_dim, self.sub_emb)
        self.fc3333 = nn.Linear(self.sub_emb, self.num_classes)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        emb1 = self.dropout(F.relu(self.fc1(x)))
        emb1 = self.dropout(F.relu(self.fc11(emb1)))
        emb1 = self.dropout(F.normalize(torch.tanh(self.fc111(emb1)), dim=1))
        y1 = self.fc1111(emb1)

        emb2 = self.dropout(F.relu(self.fc2(x)))
        emb2 = self.dropout(F.relu(self.fc22(emb2)))
        emb2 = self.dropout(F.normalize(torch.tanh(self.fc222(emb2)), dim=1))
        y2 = self.fc2222(emb2)

        emb3 = self.dropout(F.relu(self.fc3(x)))
        emb3 = self.dropout(F.relu(self.fc33(emb3)))
        emb3 = self.dropout(F.normalize(torch.tanh(self.fc333(emb3)), dim=1))
        y3 = self.fc3333(emb3)

        return emb1, emb2, emb3, y1, y2, y3

# fact model with no norm and no projection

class fact_net1(nn.Module):

    def __init__(self, input_dim, sub_emb, num_classes):
        super(fact_net1, self).__init__()
        
        self.input_dim = input_dim
        self.sub_emb = sub_emb
        self.num_classes = num_classes
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.input_dim, 4*self.input_dim) 
        self.fc11 = nn.Linear(4*self.input_dim, 4*self.input_dim)
        self.fc111 = nn.Linear(4*self.input_dim, self.sub_emb)
        self.fc1111 = nn.Linear(self.sub_emb, self.num_classes)

        self.fc2 = nn.Linear(self.input_dim, 4*self.input_dim) 
        self.fc22 = nn.Linear(4*self.input_dim, 4*self.input_dim)
        self.fc222 = nn.Linear(4*self.input_dim, self.sub_emb)
        self.fc2222 = nn.Linear(self.sub_emb, self.num_classes)

        self.fc3 = nn.Linear(self.input_dim, 4*self.input_dim) 
        self.fc33 = nn.Linear(4*self.input_dim, 4*self.input_dim)
        self.fc333 = nn.Linear(4*self.input_dim, self.sub_emb)
        self.fc3333 = nn.Linear(self.sub_emb, self.num_classes)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        emb1 = self.dropout(F.relu(self.fc1(x)))
        emb1 = self.dropout(F.relu(self.fc11(emb1)))
        emb1 = self.dropout(torch.tanh(self.fc111(emb1)))
        y1 = self.fc1111(emb1)

        emb2 = self.dropout(F.relu(self.fc2(x)))
        emb2 = self.dropout(F.relu(self.fc22(emb2)))
        emb2 = self.dropout(torch.tanh(self.fc222(emb2)))
        y2 = self.fc2222(emb2)

        emb3 = self.dropout(F.relu(self.fc3(x)))
        emb3 = self.dropout(F.relu(self.fc33(emb3)))
        emb3 = self.dropout(torch.tanh(self.fc333(emb3)))
        y3 = self.fc3333(emb3)

        return emb1, emb2, emb3, y1, y2, y3

# with proejction and norm

class fact_net3(nn.Module):

    def __init__(self, input_dim, sub_emb, num_classes):
        super(fact_net3, self).__init__()
        
        self.input_dim = input_dim
        self.sub_emb = sub_emb
        self.num_classes = num_classes
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.input_dim, 4*self.input_dim) 
        self.fc11 = nn.Linear(4*self.input_dim, 4*self.input_dim)
        self.fc111 = nn.Linear(4*self.input_dim, self.sub_emb)
        self.fc1111 = nn.Linear(self.sub_emb, self.num_classes)

        self.fc2 = nn.Linear(self.input_dim, 4*self.input_dim) 
        self.fc22 = nn.Linear(4*self.input_dim, 4*self.input_dim)
        self.fc222 = nn.Linear(4*self.input_dim, self.sub_emb)
        self.fc2222 = nn.Linear(self.sub_emb, self.num_classes)

        self.fc3 = nn.Linear(self.input_dim, 4*self.input_dim) 
        self.fc33 = nn.Linear(4*self.input_dim, 4*self.input_dim)
        self.fc333 = nn.Linear(4*self.input_dim, self.sub_emb)
        self.fc3333 = nn.Linear(self.sub_emb, self.num_classes)

        self.fc_projection = nn.Linear(self.sub_emb, self.input_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        emb1 = self.dropout(F.relu(self.fc1(x)))
        emb1 = self.dropout(F.relu(self.fc11(emb1)))
        emb1 = self.dropout(F.normalize(torch.tanh(self.fc111(emb1)), dim=1))
        y1 = self.fc1111(emb1)

        emb2 = self.dropout(F.relu(self.fc2(x)))
        emb2 = self.dropout(F.relu(self.fc22(emb2)))
        emb2 = self.dropout(F.normalize(torch.tanh(self.fc222(emb2)), dim=1))
        y2 = self.fc2222(emb2)

        emb3 = self.dropout(F.relu(self.fc3(x)))
        emb3 = self.dropout(F.relu(self.fc33(emb3)))
        emb3 = self.dropout(F.normalize(torch.tanh(self.fc333(emb3)), dim=1))
        y3 = self.fc3333(emb3)
        
        sum_emb = emb1 + emb2 + emb3
        x_recon = F.normalize(torch.tanh(self.fc_projection(sum_emb)))
        return emb1, emb2, emb3, y1, y2, y3, x_recon