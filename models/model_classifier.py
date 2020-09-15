import torch
import torch.nn as nn
import torch.nn.functional as F

act_layer = {'elu':     nn.ELU(),
             'lrelu':   nn.LeakyReLU(),
             'relu':    nn.ReLU(),
             'prelu':   nn.PReLU(),
             'relu6':   nn.ReLU6(),
             'gelu':    nn.GELU(),
             'tanh':    nn.Tanh,
             'sigmoid': nn.Sigmoid(),
             }
norm_layer = {'bn':     nn.BatchNorm1d}

# TODO: should we add dropout?
# cls_1: a very simple MLP classifier
class cls_1(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim, ):
        super(cls_1, self).__init__()

        self.dense0 = nn.Linear(in_features=input_dim, out_features=h_sizes[0])
        self.bn0    = nn.BatchNorm1d(num_features=h_sizes[0])
        self.dense1 = nn.Linear(in_features=h_sizes[0], out_features=h_sizes[1])
        self.bn1    = nn.BatchNorm1d(num_features=h_sizes[1])
        self.dense2 = nn.Linear(in_features=h_sizes[1], out_features=output_dim)
        
    
    def forward(self, x, dropout=0):
        x = self.dense0(x)
        x = self.bn0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=dropout)
        x = self.dense1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=dropout)
        x = self.dense2(x)
        return x


# cls_1: using simple position encoder + MLP as a classifier
class cls_2(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim, 
                 max_position_embeddings, nhead=8,
                 act_layers=[], norm_layers=[]):
        super(cls_2, self).__init__()

        self.hiddens = nn.ModuleList()
        self.position_encode = nn.Embedding(max_position_embeddings, 1)

        h_sizes = [input_dim] + h_sizes + [output_dim]
        for i in range(len(h_sizes) - 1):
            self.hiddens.append(nn.Linear(h_sizes[i], h_sizes[i+1]))
            if i < len(act_layers) and i < len(h_sizes) - 2:
                if act_layers[i] in act_layer:
                    self.hiddens.append(act_layer[act_layers[i]])
            if i < len(norm_layers) and i < len(h_sizes) - 2:
                if norm_layers[i] in norm_layer:
                    self.hiddens.append(norm_layer[norm_layers[i]](h_sizes[i+1]))
    
    def forward(self, x, dropout=0):
        position_id = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        position_encode = self.position_encode(position_id).squeeze(-1)
        x += position_encode
        for i, l in enumerate(self.hiddens):
            x = l(x)
            x = F.dropout(x, p=dropout)
        return x

# for debug
if __name__ == '__main__':
    mlp = cls_1(200, [100,75], 2)
    x = torch.rand(32, 200)
    # print('result:', mlp(x))
    print('shape:', mlp(x).shape)