import torch
import torch.nn as nn
import torch.nn.functional as F

class fact_net(nn.Module):

    def __init__(self, input_dim, sub_emb, num_classes, proj):
        super(fact_net, self).__init__()
        
        self.scale = 4
        self.input_dim = input_dim
        self.sub_emb = sub_emb
        self.num_classes = num_classes
        self.proj = proj
        if self.proj:
            print("loaded model with projection layer")
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.input_dim, self.scale*self.input_dim) 
        self.fc11 = nn.Linear(self.scale*self.input_dim, self.scale*self.input_dim)
        self.fc111 = nn.Linear(self.scale*self.input_dim, self.sub_emb)
        self.fc1111 = nn.Linear(self.sub_emb, self.num_classes)

        self.fc2 = nn.Linear(self.input_dim, self.scale*self.input_dim) 
        self.fc22 = nn.Linear(self.scale*self.input_dim, self.scale*self.input_dim)
        self.fc222 = nn.Linear(self.scale*self.input_dim, self.sub_emb)
        self.fc2222 = nn.Linear(self.sub_emb, self.num_classes)

        self.fc3 = nn.Linear(self.input_dim, self.scale*self.input_dim) 
        self.fc33 = nn.Linear(self.scale*self.input_dim, self.scale*self.input_dim)
        self.fc333 = nn.Linear(self.scale*self.input_dim, self.sub_emb)
        self.fc3333 = nn.Linear(self.sub_emb, self.num_classes)

        if self.proj:
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
        if self.proj==True:
            x_recon = F.normalize(torch.tanh(self.fc_projection(sum_emb)),dim=1)
        else:
            x_recon = F.normalize(sum_emb, dim=1)
        return emb1, emb2, emb3, y1, y2, y3, x_recon