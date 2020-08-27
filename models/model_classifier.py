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
    def __init__(self, input_dim, h_sizes, output_dim, 
                 act_layers=[], norm_layers=[], dropouts=[]):
        super(cls_1, self).__init__()

        self.hiddens = nn.ModuleList()
        self.dropouts = dropouts
        h_sizes = [input_dim] + h_sizes + [output_dim]
        for i in range(len(h_sizes) - 1):
            self.hiddens.append(nn.Linear(h_sizes[i], h_sizes[i+1]))
            if i < len(act_layers) and i < len(h_sizes) - 2:
                if act_layers[i] in act_layer:
                    self.hiddens.append(act_layer[act_layers[i]])
            if i < len(norm_layers) and i < len(h_sizes) - 2:
                if norm_layers[i] in norm_layer:
                    self.hiddens.append(norm_layer[norm_layers[i]](h_sizes[i+1]))
            if i < len(dropouts) and i < len(h_sizes) - 2:
                if dropouts[i] > 0:
                    self.hiddens.append(nn.Dropout(p=dropouts[i]))
    
    def forward(self, x):
        for i, l in enumerate(self.hiddens):
            x = l(x)
        return x


# for debug
if __name__ == '__main__':
    mlp = cls_1(10, [100,75,40], 2, 
                act_layers=['gelu', 'gelu', 'gelu'],
                norm_layers=['bn', 'bn', 'bn'],
                dropouts = [0.5, 0.5, 0.5],
                )
    x = torch.rand(13, 10)
    print(mlp)
    print('result:', mlp(x))
    print('shape:', mlp(x).shape)