# coding:utf-8
"""
Model definitions for compound virtual screening based on ligand properties
Created  :   6, 11, 2019
Revised  :   6, 11, 2019
Author   :  David Leon (dawei.leng@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""

__author__ = 'dawei.leng'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_ext.module import BatchNorm1d
import numpy as np

__all__ = [
    'model_0',
    'model_1',
    'model_4v1'
]

#--- utility modules ---#
def neighbor_op(x, padded_neighbors, op='max', include_self=False):
    """
    Generic neighborhood operations
    :param x: node features, (N, D).
    :param padded_neighbors: (N, max_degree_in_batch), int64, each row represents the 1-hop neighbors for node_i; `-1`
                             are padded to indicate invalid values, self nodes are presumed excluded.
    :param include_self: whether append self nodes when perform neighborhood operations. If `padded_neighbors` already
                         include self nodes, remember set this flag to False.
    :param op: supported neighborhood operations include {'max', 'min', 'sum', 'mean'}.
               when 'mean', possible `inf` values will be replaced with 0;
    :return:
    """
    if include_self:
        N = x.shape[0]
        self_idxs = torch.arange(0, N, dtype=padded_neighbors.dtype, device=padded_neighbors.device).view(N, 1)
        padded_neighbors = torch.cat([self_idxs, padded_neighbors], dim=1)

    if op == 'max':
        dummy        = torch.min(x, dim=0)[0]                  # (D,)
        x_with_dummy = torch.cat([x, dummy.reshape((1, -1))])  # (N+1, D)
        x_neighbors  = x_with_dummy[padded_neighbors]          # (N, max_degree_in_batch, D)
        result       = torch.max(x_neighbors, dim=1)[0]        # (N, D)
    elif op == 'min':
        dummy        = torch.max(x, dim=0)[0]                  # (D,)
        x_with_dummy = torch.cat([x, dummy.reshape((1, -1))])  # (N+1, D)
        x_neighbors  = x_with_dummy[padded_neighbors]          # (N, max_degree_in_batch, D)
        result       = torch.min(x_neighbors, dim=1)[0]        # (N, D)
    elif op in {'sum', 'mean'}:
        D            = x.shape[1]
        dummy        = x.new_zeros(size=(1, D))
        x_with_dummy = torch.cat([x, dummy], dim=0)            # (N+1, D)
        x_neighbors  = x_with_dummy[padded_neighbors]          # (N, max_degree_in_batch, D)
        result       = torch.sum(x_neighbors, dim=1)           # (N, D)
        if op == 'mean':
            mask   = padded_neighbors >= 0
            nums   = mask.sum(dim=1).view(-1, 1)
            result = result / nums
            result[result == float('inf')] = 0.0
    else:
        raise ValueError('Invalid op = %s' % op)
    return result

def graph_maxpool_DeepChem(node_features, deg_slice, deg_adj_list, min_degree=0, max_degree=10):
    """
    graph max pooling function refactored from `tensorflow_version.ligandbasedpackage.model_ops.GraphPool`
    [DV] blending center node and neighboring nodes with max(), no much sense
    :param node_features: 2D tensor with shape (node_num, feature_dim)
    :param deg_slice:     2D tensor with shape (max_deg+1-min_deg,2, 2)
    :param deg_adj_list: list of 2D tensor with shape (node_num, degree_num), len = max_deg+1-min_deg
    :param min_degree: int, 0 or 1
    :param max_degree: int
    :return:
    """
    # maxed_node_feature_list = (max_degree + 1 - min_degree) * [None]
    maxed_node_feature_list = []


    degree_dim = deg_slice.shape[0]    # max_degree + 1, [0, max_degree]
    # for deg in range(1, max_degree+1):
    for deg in range(min_degree, min(max_degree+1, degree_dim)):
        if deg == 0:
            self_node_feature = node_features[deg_slice[0, 0]:deg_slice[0, 0] + deg_slice[0, 1], :]  # shape = (n_node_with_given_degree, feature_dim)
            maxed_node_feature_list.append(self_node_feature)
        else:
            if deg_slice[deg, 1] > 0:   # [:,0] for starting index, [:,1] for span size
                start_idx = deg_slice[deg, 0]
                end_idx   = deg_slice[deg, 0] + deg_slice[deg, 1]
                self_node_feature = node_features[start_idx:end_idx, :]  # shape = (n_node_with_given_degree, feature_dim)
                self_node_feature = self_node_feature[:, None, :]   # shape = (n_node_with_given_degree, 1, feature_dim)
                neighbor_node_features = node_features[deg_adj_list[deg-1],:]  # shape = (n_node_with_given_degree, degree, feature_dim)
                 # neighbor_node_features = torch.gather(node_features, dim=0, index=deg_adj_list[deg - 1])  # shape = (n_node_with_given_degree, degree, feature_dim)
                tmp_node_features = torch.cat([self_node_feature, neighbor_node_features], dim=1)  # shape = (n_node_with_given_degree, degree+1, feature_dim)
                maxed_node_feature, _ = torch.max(tmp_node_features, dim=1, keepdim=False)  # shape = (n_node_with_given_degree, feature_dim)
                # maxed_node_feature_list[deg - min_degree] = maxed_node_feature
                maxed_node_feature_list.append(maxed_node_feature)

    result = torch.cat(maxed_node_feature_list, dim=0) # todo: pytorch does not handle None properly.
    return result

def graph_gather_DeepChem(node_features, membership, n_graph=None):
    """
    [DV] Retrieve atoms for each molecule by `membership` and aggregate atoms features for each molecule to form the final
    graph representation. Still no much sense.
    Implementation difference: 1) no activation affiliated
    :param node_features: 2D tensor with shape (node_num, feature_dim)
    :param membership:    1D tensor with shape (node_num,)
    :param n_graph: int, how many graphs involved in `node_features`
    :return: graph_features, 2D tensor with shape (n_graph, 2 * feature_dim)
    """
    if n_graph is None:
        n_graph = int(max(membership))+1
    node_features_for_each_graph = []
    for i in range(n_graph):
        mask = membership == i
        node_features_for_each_graph.append(node_features[mask, :])

    mean_feature_for_each_graph = [
        torch.mean(item, dim=0, keepdim=True)
        for item in node_features_for_each_graph
        ]

    max_feature_for_each_graph = [
        torch.max(item, dim=0, keepdim=True)[0]
        for item in node_features_for_each_graph
        ]
    mean_features_graph = torch.cat(mean_feature_for_each_graph, dim=0)
    max_features_graph  = torch.cat(max_feature_for_each_graph,  dim=0)
    graph_features = torch.cat([mean_features_graph, max_features_graph], dim=1)  # (n_graph, 2 * feature_dim)

    return graph_features

class Graph_Conv_DeepChem(nn.Module):
    """
    Graph convolution module refactored from `ligandbasedpackage.model_ops.GraphConv`
    [DV] there is no much sense for such degree-wise "conv" here, highly doubt its performance
    Implementation difference: 1) no activation affiliated
    """
    def __init__(self,
                 in_channels,
                 output_dim,
                 min_deg=0,
                 max_deg=10):
        super().__init__()
        self.in_channels       = in_channels           # input feature dimension
        self.output_dim        = output_dim          # output feature dimension
        self.min_degree        = min_deg
        self.max_degree        = max_deg
        self.param_tuple_size  = 2 * max_deg + (1 - min_deg)

        self.W_list = [Parameter(torch.empty(in_channels, output_dim)) for _ in range(self.param_tuple_size)]
        self.b_list = [Parameter(torch.empty(output_dim)) for _ in range(self.param_tuple_size)]
        #todo: pytorch sucks here because it cannot name the model parameter automatically, needs API redesign, dandelion_torch would be necessary
        for i, W in enumerate(self.W_list):
            self.register_parameter('W%d'%i, W)
        for i, b in enumerate(self.b_list):
            self.register_parameter('b%d'%i, b)
        self.reset_parameters()
        self.predict = self.forward

    def reset_parameters(self, W_init=nn.init.xavier_uniform_, b_init=nn.init.zeros_):
        for W in self.W_list:
            W_init(W)
        for b in self.b_list:
            b_init(b)

    def forward(self, node_features, deg_slice, deg_adj_list):
        """

        :param node_features: (node_num, feature_dim),
        :param deg_slice: (max_deg+1-min_deg,2, 2),
        :param deg_adj_list: list of tensor with shape=(node_num, degree_num), len = max_deg+1-min_deg
        :return:
        """
        W, b = iter(self.W_list), iter(self.b_list)

        # [DV] aggregate neighbors at each degree level, returned a list of len=self.max_degree
        deg_summed = self._sum_neigh(node_features, deg_adj_list)

        # Get collection of modified atom features
        degree_dim = deg_slice.shape[0]  # max_degree + 1, [0, max_degree]
        new_node_feature_collection = []
        # for deg in range(min(self.max_degree + 1, degree_dim)):
        for deg in range(degree_dim):
            if deg == 0:
                self_node_feature = node_features[deg_slice[0, 0]:deg_slice[0, 0] + deg_slice[0, 1], :]  # shape = (n_node_with_given_degree, feature_dim)
                out = torch.matmul(self_node_feature, next(W)) + next(b)
                new_node_feature_collection.append(out)
            else:
        # for deg in range(1, min(self.max_degree + 1, degree_dim)):      # [DV] todo: shouldn't we start from `min_degree` here?
                neighbour_feature = deg_summed[deg - 1]    # [DV] shape = (n_node_with_given_degree, feature_dim)
                self_node_feature = node_features[deg_slice[deg, 0]:deg_slice[deg, 0]+deg_slice[deg, 1], :]  # [DV] shape = (n_node_with_given_degree, feature_dim)
                # Apply hidden affine to relevant atoms and append
                # [DV] todo: 1) using different affine transforms for `self_atoms` and `rel_atoms` does NOT make any sense
                # [DV] todo: 2) using different affine transforms for different degree graph nodes does NOT make any sense
                # [DV] todo: 3) the only sensible way is to `concat`, not `+` for `self_atoms` and `rel_atoms`
                neighbour_feature = torch.matmul(neighbour_feature, next(W)) + next(b)
                self_node_feature = torch.matmul(self_node_feature, next(W)) + next(b)
                out = neighbour_feature + self_node_feature
                new_node_feature_collection.append(out)

        # if self.min_degree == 0:
        #     self_node_feature = node_features[deg_slice[0, 0]:deg_slice[0, 0]+deg_slice[0, 1], :]  # shape = (n_node_with_given_degree, feature_dim)
        #     out               = torch.matmul(self_node_feature, next(W)) + next(b)
        #     new_node_feature_collection[0] = out

        # Combine all atoms back into the list
        node_features = torch.cat(new_node_feature_collection, dim=0)

        return node_features

    def _sum_neigh(self, node_features, deg_adj_list):
        """
        [DV] todo: what's the rationale of dividing graph nodes by degree?
        :param node_features: (node_num, feature_dim)
        :param deg_adj_list: list of element with shape (node_num, degree_num)
        :return:
        """
        """Store the summed atoms by degree"""
        list_size = len(deg_adj_list)
        deg_summed = list_size * [None]
        for deg in range(list_size):   # [DV] todo: shoudn't we start from `min_degree` here?
            # neighbor_node_features = torch.gather(node_features, dim=0, index=deg_adj_list[deg - 1])  # [DV] shape = (n_node_with_given_degree, degree, feature_dim)
            neighbor_node_features = node_features[deg_adj_list[deg]]  # [DV] shape = (n_node_with_given_degree, degree, feature_dim)
            summed_atoms = torch.sum(neighbor_node_features, dim=1, keepdim=False)   # [DV] shape = (n_node_with_given_degree, feature_dim)
            deg_summed[deg] = summed_atoms
        return deg_summed    # len = self.max_degree

#---- DeepChem's reference model
class model_0(nn.Module):
    """
    Same model with `ligandbasedpackage.graph_models.GraphConvTensorGraph`
    """
    def __init__(self,
                 num_embedding=0,
                 feature_dim=75,
                 graph_conv_layer_size=(64, 64),
                 dense_layer_size=128,
                 dropout=0.0,
                 output_dim=2,
                 min_degree=0,
                 max_degree=10,
                 **kwargs):
        """

        :param graph_conv_layers:
        :param dense_layer_size:
        :param dropout:
        :param output_dim:
        """
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=feature_dim)
        self.graph_conv_layer_size = graph_conv_layer_size
        self.dense_layer_size      = dense_layer_size
        self.dropout               = dropout
        self.output_dim            = output_dim
        self.min_degree            = min_degree
        self.max_degree            = max_degree
        self.error_bars            = True if 'error_bars' in kwargs and kwargs['error_bars'] else False
        self.dense0                = nn.Linear(in_features=graph_conv_layer_size[1], out_features=dense_layer_size)
        self.dense1                = nn.Linear(in_features=2*dense_layer_size, out_features=output_dim)
        self.gconv0                = Graph_Conv_DeepChem(in_channels=feature_dim, output_dim=graph_conv_layer_size[0], min_deg=self.min_degree, max_deg=self.max_degree)
        self.gconv1                = Graph_Conv_DeepChem(in_channels=graph_conv_layer_size[0], output_dim=graph_conv_layer_size[1], min_deg=self.min_degree, max_deg=self.max_degree)
        self.bn0                   = nn.BatchNorm1d(num_features=graph_conv_layer_size[0])
        self.bn1                   = nn.BatchNorm1d(num_features=graph_conv_layer_size[1])
        self.bn2                   = nn.BatchNorm1d(num_features=dense_layer_size)


    def forward(self, x, degree_slice, membership, deg_adj_list):
        """
        Forward pass
        :param x: either feature matrix of float tensor(node_num, feature_dim), or node index tensor (node_num,)
        :param degree_slice: int tensor matrix, (max_deg+1-min_deg, 2)
        :param membership: int tensor, (node_num,)
        :param deg_adj_list: list of int tensor with shape=(node_num, degree_num), len = max_deg+1-min_deg
        :return: un-normalized class distribution
        """
        if self.num_embedding > 0:
            x = self.emb0.forward(x)     # (node_num, ) -> (node_num, feature_dim)
        x = self.gconv0(x, degree_slice, deg_adj_list)
        x = torch.relu(x)
        x = self.bn0.forward(x)
        x = graph_maxpool_DeepChem(x, degree_slice, deg_adj_list, min_degree=self.min_degree, max_degree=self.max_degree)
        x = self.gconv1(x, degree_slice, deg_adj_list)
        x = torch.relu(x)
        x = self.bn1.forward(x)
        x = graph_maxpool_DeepChem(x, degree_slice, deg_adj_list, min_degree=self.min_degree, max_degree=self.max_degree)
        x = self.dense0.forward(x)
        x = self.bn2.forward(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = graph_gather_DeepChem(x, membership)
        x = torch.tanh(x)
        x = self.dense1.forward(x)
        # x = torch.log_softmax(x, dim=1)
        return x


    def predict(self, *inputs):
        self.train(False)
        return self.forward(*inputs)

#---- RNN baseline
class model_1(nn.Module):
    """
    A basic RNN model served as baseline
    """
    def __init__(self,
                 num_embedding=0,
                 feature_dim=75,
                 hidden_size=128,
                 bidirectional=True,
                 dropout=0.5,
                 output_dim=2):
        """

        :param num_embedding: if >0, input is assumed to be SMILES string; else input will be assumed to be atom feature sequence
        :param feature_dim:
        :param hidden_size:
        :param num_layers:
        :param bidirectional:
        :param dropout:
        :param output_dim:
        """
        super().__init__()
        self.num_embedding = num_embedding
        num_directions = 2 if bidirectional else 1
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=feature_dim)
        self.lstm0  = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=1,
                             bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.lstm1  = nn.LSTM(input_size=num_directions*hidden_size, hidden_size=hidden_size, num_layers=1,
                             bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.lstm2  = nn.LSTM(input_size=num_directions*hidden_size, hidden_size=hidden_size, num_layers=1,
                             bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.dense0 = nn.Linear(in_features=num_directions*hidden_size, out_features=output_dim)

    def forward(self, x):
        """
        :param x: (B, T) if self.num_embedding > 0 else (B, T, D)
        :return:
        """
        if self.num_embedding > 0:
            x = self.emb0.forward(x)
        x, _ = self.lstm0.forward(x)    # (B, T, num_direction*hidden_size)
        x = torch.transpose(x, 1, 2)    # (B, T, D)->(B, D, T)
        x = F.max_pool1d(x, kernel_size=2)  # (B, D, T) -> (B, D, T//2)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm1.forward(x)
        x = torch.transpose(x, 1, 2)    # (B, T, D)->(B, D, T)
        x = F.max_pool1d(x, kernel_size=2)  # (B, D, T) -> (B, D, T//2)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm2.forward(x)
        x = x[:, -1, :]                 # (B, D)
        x = self.dense0.forward(x)
        return x

#---- GIN model
def max_pooling_GIN(x, padded_neighbors):
    """
    :param x: (node_num, D)
    :param padded_neighbors: (node_num, max_degree), int64, each row represents the 1-hop neighbors for node_i; `-1`
                             are padded to indicate invalid values.
    :return:
    """
    dummy = torch.min(x, dim=0)[0]   # (D,)
    h_with_dummy = torch.cat([x, dummy.reshape((1, -1))])  # (node_num+1, D)
    result = torch.max(h_with_dummy[padded_neighbors], dim=1)[0]  # (node_num, max_degree, D) -> (node_num, D), `-1` in `padded_neighbors` indexs the last row of `h_with_dummy`
    return result

class GIN_block(nn.Module):
    """
    Building block of graph isomorphism network
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 neighbor_pooling_method='sum',  # equals to the so called "GCN"
                 layer_num=2,
                 eps=None
                 ):
        super().__init__()
        self.input_dim      = input_dim
        self.output_dim     = output_dim
        self.pooling_method = neighbor_pooling_method.lower()
        if eps is None:
            self.eps = None
        else:
            self.eps        = Parameter(torch.scalar_tensor(eps, dtype=torch.float32))        # to be learned
        self.layer_num      = layer_num
        self.linears        = nn.ModuleList()
        self.bns            = nn.ModuleList()
        if self.pooling_method not in {'sum', 'mean', 'max'}:
            raise ValueError("neighbor_pooling_method should be in {'sum', 'mean', 'max'}")
        in_features = input_dim
        for i in range(self.layer_num):
            self.linears.append(nn.Linear(in_features=in_features, out_features=output_dim))
            self.bns.append(nn.BatchNorm1d(num_features=output_dim))
            in_features = output_dim

    def forward(self, x, padded_neighbors=None, adj_matrix=None):
        """
        :param x: (node_num, D)
        :param padded_neighbors:
        :param adj_matrix: (node_num, node_num), adjacency matrix, sparse or dense
        :return:
        """
        if self.pooling_method == 'max':
            pooled = max_pooling_GIN(x, padded_neighbors)
        else:
            pooled = torch.spmm(adj_matrix, x)
            if self.pooling_method == 'mean':
                degree = torch.spmm(adj_matrix, torch.ones((adj_matrix.shape[0], 1)).to(adj_matrix.device))
                pooled = pooled / degree
        if self.eps is None:
            pooled = pooled + x
        else:
            pooled = pooled + (1 + self.eps) * x         # self-connection
        x = pooled
        for i in range(self.layer_num):
            x = self.linears[i].forward(x)
            x = self.bns[i].forward(x)
            x = F.relu(x)
        return x

class model_GIN(nn.Module):
    """
    GIN: Graph Isomorphism Network as described in Ref.1
    Ref [1]: How powerful are graph neural networks? Keyulu Xu, Weihua Hu, etc., arxiv 2019
    """
    def __init__(self,
                 num_embedding=0,
                 block_num=4,
                 block_layer_num=2,
                 input_dim=75,
                 hidden_dim=64,
                 output_dim=None,     # class num
                 final_dropout=0.5,
                 neighbor_pooling_method='sum',
                 readout_method='sum',
                 eps=0.0,             # set to None to disable learning this parameter
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
        self.block_num = block_num
        self.final_dropout = final_dropout
        self.neighbor_pooling_method = neighbor_pooling_method
        self.readout_method = readout_method
        if self.readout_method not in {'sum', 'mean'}:
            raise ValueError("read_out_method should be in {'sum', 'mean'}")
        self.blocks  = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_features=input_dim, out_features=output_dim))
        block_input_dim = input_dim
        for i in range(self.block_num):
            self.blocks.append(GIN_block(input_dim=block_input_dim, output_dim=hidden_dim,
                                         neighbor_pooling_method=neighbor_pooling_method,
                                         layer_num=block_layer_num, eps=eps))
            self.linears.append(nn.Linear(in_features=hidden_dim, out_features=output_dim))
            block_input_dim = hidden_dim


    def forward(self, x, padded_neighbors=None, adj_matrix=None, membership=None):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param padded_neighbors: (node_num, max_degree), int64, each row represents the 1-hop neighbors for node_i;
                                `-1` are padded to indicate invalid values.
        :param adj_matrix: (node_num, node_num), adjacency matrix, sparse
        :param membership: (batch_size, node_num) with element `1` indicating to which graph a given node belongs
        :return:
        """
        if self.num_embedding > 0:
            x = self.emb0.forward(x)
        #--- aggregation ---#
        hiddens = [x]
        for i in range(self.block_num):
            x = self.blocks[i].forward(x=x, padded_neighbors=padded_neighbors, adj_matrix=adj_matrix)
            hiddens.append(x)
        #--- readout ---#
        graph_representation = 0
        for i in range(self.block_num+1):
            pooled = torch.spmm(membership, hiddens[i])
            if self.readout_method == 'mean':
                node_num = torch.spmm(membership, torch.ones((membership.shape[1], 1)).to(membership.device))
                pooled = pooled / node_num
            graph_representation += F.dropout(self.linears[i].forward(pooled), self.final_dropout, training=self.training)

        return graph_representation

model_2 = model_GIN

#---- Graph Unet model

class Graph_Conv_Block_A0(nn.Module):
    """
    Graph convolution operation defined in GUNet paper,
    it's an over-simplified version of spectral graph convolution
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, A, x, dropout=0.1):
        """
        :param A: adjacency matrix with self-connection, (node_num, node_num), can be sparse or dense
        :param x: node feature matrix, (node_num, feature_dim)
        :param dropout:
        :return: x <- AxW + bias
        """
        x = F.dropout(x, p=dropout, training=self.training)
        x = torch.spmm(A, x)     # returned x is dense, A can be sparse or dense
        x = self.dense(x)        # (node_num, input_dim) -> (node_num, output_dim), no activation affiliated
        return x

class Graph_Pool(nn.Module):
    """
    Graph pooling by linear projection, as described in GraphUNet paper
    """
    def __init__(self, input_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, 1)

    def forward(self, A, x, k):
        """
        :param A: adjacency matrix, (node_num, node_num)
        :param x: node feature matrix, (node_num, feature_dim)
        :param k: int >=1, or float in (0, 1.0]
        :return:
        """
        proj_scores = torch.sigmoid(self.dense(x))   # 1D linear projection, (node_num, 1)
        # proj_scores = torch.sigmoid(self.dense(x)/100)   # 1D linear projection, (node_num, 1), as in GUNet's official code
        proj_scores = torch.squeeze(proj_scores)
        node_num    = proj_scores.shape[0]
        if not isinstance(k, int):    # should be float in (0, 1.0]
            k = max(1, k * node_num)  # at least 1
        vs, idxs = torch.topk(proj_scores, int(k))
        x_pooled = x[idxs, :]         # (k, feature_dim)
        vs       = torch.unsqueeze(vs, -1)  # (k, 1)
        x_pooled = x_pooled * vs      # re-weighting
        A        = A * A
        A_pooled = A[idxs,:][:, idxs] # todo: be compatible with sparse A
        return A_pooled, x_pooled, idxs

class Graph_Unpool(nn.Module):
    """
    Graph unpooling by zero-padding, as described in GraphUNet paper
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, node_num, idxs):
        feature_dim = x.shape[1]
        x_unpooled = torch.zeros([node_num, feature_dim]).to(x.device)
        x_unpooled[idxs, :] = x
        return x_unpooled

class model_GUNet(nn.Module):
    """
    This implementation is basically the same with the official implementation, which is only applicable
    for node feature embedding/reconstruction.
    The original graph UNet does not consider adjacency matrix reconstruction as optimization target.
    """
    def __init__(self,
                 input_dim=None,
                 hidden_dim=48,            # default value  the same as in GUNet paper
                 output_dim=None,          # class_num
                 ks=(0.9, 0.7, 0.6, 0.5),  # default values the same as in GUNet paper
                 dropout=0.3               # default value  the same as in GUNet paper
                 ):
        super().__init__()
        self.ks = ks
        self.start_gcn  = Graph_Conv_Block_A0(input_dim, hidden_dim)
        self.bottom_gcn = Graph_Conv_Block_A0(hidden_dim, hidden_dim)
        self.end_gcn    = Graph_Conv_Block_A0(2 * hidden_dim, output_dim)
        self.down_gcns  = nn.ModuleList()
        self.up_gcns    = nn.ModuleList()
        self.pools      = nn.ModuleList()
        self.unpools    = nn.ModuleList()
        self.levels     = len(ks)
        self.dropout    = dropout
        for i in range(self.levels):
            self.down_gcns.append(Graph_Conv_Block_A0(hidden_dim, hidden_dim))
            self.up_gcns.append(Graph_Conv_Block_A0(hidden_dim, hidden_dim))
            self.pools.append(Graph_Pool(hidden_dim))
            self.unpools.append(Graph_Unpool())

    def forward(self, A, x):
        """
        :param A: normalized adjacency matrix, (node_num, node_num), dense matrix
        :param x: node feature matrix, (node_num, input_dim)
        :return: x: restored node feature matrix, (node_num, output_dim)
        """
        A_list        = []                 # [DV] adjacency matrices
        idxs_list     = []
        down_gcn_outs = []
        x = self.start_gcn(A, x, dropout=self.dropout)
        org_x = x
        #--- down-pooling ---#
        for i in range(self.levels):
            x = self.down_gcns[i](A, x, dropout=self.dropout)
            down_gcn_outs.append(x)
            A_list.append(A)
            A, x, idxs = self.pools[i](A, x, self.ks[i])
            idxs_list.append(idxs)
        x = self.bottom_gcn(A, x, dropout=self.dropout)
        #--- up-pooling---#
        for i in range(self.levels):
            up_idx = self.levels - i - 1
            A, idxs = A_list[up_idx], idxs_list[up_idx]
            x = self.unpools[i](x, A.shape[0], idxs)
            x = self.up_gcns[i](A, x, dropout=self.dropout)
            x = x.add(down_gcn_outs[up_idx])  # [DV] `add` instead of `concat` is used in GUNet paper
        x = torch.cat([x, org_x], 1)          # [DV] `concat` is used in GUNet paper
        x = self.end_gcn(A, x, dropout=self.dropout)
        return x

class model_3(nn.Module):
    """
    Graph classification using GUNet as backbone for graph node representation learning as described in GUNet paper.
    Implementation is basically the same with the original paper except for the readout function.
    """
    def __init__(self,
                 num_embedding=0,
                 ks=(0.9, 0.7, 0.6, 0.5),
                 input_dim=75,
                 hidden_dim=64,
                 output_dim=None,     # class num
                 dropout=0.1,
                 readout_method='sum'
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
        self.gunet = model_GUNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                                 ks=ks, dropout=dropout)
        self.readout_method = readout_method
        if self.readout_method not in {'sum', 'mean'}:
            raise ValueError("read_out_method should be in {'sum', 'mean'}")
        self.dense = nn.Linear(in_features=hidden_dim, out_features=output_dim)


    def forward(self, x, adj_matrix, membership=None):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, input_dim) else
        :param adj_matrix: (node_num, node_num)
        :param membership: (batch_size, node_num) with element `1` indicating to which graph a given node belongs
        :return:
        """
        if self.num_embedding > 0:
            x = self.emb0.forward(x)
        #--- aggregation ---#
        x = self.gunet(adj_matrix, x)

        #--- readout ---#
        pooled = torch.spmm(membership, x)      # sum
        if self.readout_method == 'mean':
            node_num = torch.spmm(membership, torch.ones((membership.shape[1], 1)).to(membership.device))
            pooled = pooled / node_num
        graph_representation = self.dense.forward(pooled)

        return graph_representation

class model_GUNet_v1(nn.Module):
    """
    GUNet variation version 1
    """
    def __init__(self,
                 input_dim=None,
                 hidden_dim=48,            # default value  the same as in GUNet paper
                 output_dim=None,          # class_num
                 ks=(0.9, 0.7, 0.6, 0.5),  # default values the same as in GUNet paper
                 dropout=0.3               # default value  the same as in GUNet paper
                 ):
        super().__init__()
        self.ks = ks
        self.start_gcn  = Graph_Conv_Block_A0(input_dim, hidden_dim)
        self.bottom_gcn = Graph_Conv_Block_A0(hidden_dim, 2*hidden_dim)
        self.end_gcn    = Graph_Conv_Block_A0(3 * hidden_dim, output_dim)
        self.down_gcns  = nn.ModuleList()
        self.up_gcns    = nn.ModuleList()
        self.pools      = nn.ModuleList()
        self.unpools    = nn.ModuleList()
        self.levels     = len(ks)
        self.dropout    = dropout
        for i in range(self.levels):
            self.down_gcns.append(Graph_Conv_Block_A0(hidden_dim, hidden_dim))
            self.up_gcns.append(Graph_Conv_Block_A0(2*hidden_dim, hidden_dim))
            self.pools.append(Graph_Pool(hidden_dim))
            self.unpools.append(Graph_Unpool())

    def forward(self, A, x, membership=None):
        """
        :param A: normalized adjacency matrix, (node_num, node_num), dense matrix
        :param x: node feature matrix, (node_num, input_dim)
        :return: x: restored node feature matrix, (node_num, output_dim)
        """
        A_list        = []                 # [DV] adjacency matrices
        idxs_list     = []
        down_gcn_outs = []
        x = self.start_gcn(A, x, dropout=self.dropout)
        org_x = x
        #--- down-pooling ---#
        for i in range(self.levels):
            x = self.down_gcns[i](A, x, dropout=self.dropout)
            down_gcn_outs.append(x)
            A_list.append(A)
            A, x, idxs = self.pools[i](A, x, self.ks[i])
            idxs_list.append(idxs)
            if membership is not None:
                membership = membership[:, idxs]
        bottom_x = self.bottom_gcn(A, x, dropout=self.dropout)
        #--- up-pooling---#
        x = bottom_x
        for i in range(self.levels):
            up_idx = self.levels - i - 1
            A, idxs = A_list[up_idx], idxs_list[up_idx]
            x = self.unpools[i](x, A.shape[0], idxs)
            x = self.up_gcns[i](A, x, dropout=self.dropout)
            # x = x.add(down_gcn_outs[up_idx])  # [DV] `add` instead of `concat` is used in GUNet paper
            x = torch.cat([x,down_gcn_outs[up_idx]], dim=1)  # change to concat
        x = torch.cat([x, org_x], dim=1)          # [DV] `concat` is used in GUNet paper
        x = self.end_gcn(A, x, dropout=self.dropout)
        return x, bottom_x, membership

class model_3v1(nn.Module):
    """
    """
    def __init__(self,
                 num_embedding=0,
                 ks=(0.9, 0.7, 0.6, 0.5),
                 input_dim=75,
                 hidden_dim=64,
                 output_dim=None,     # class num
                 dropout=0.1,
                 readout_method='sum',
                 channels = 32
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        self.dropout = dropout
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
            self.decode = nn.Linear(in_features=input_dim, out_features=num_embedding)
            self.decode.weight = self.emb0.weight
        self.GUNs = nn.ModuleList()
        self.channels = channels
        for i in range(self.channels):
            gunet = model_GUNet_v1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, ks=ks, dropout=dropout)
            self.GUNs.append(gunet)
        self.readout_method = readout_method
        if self.readout_method not in {'sum', 'mean'}:
            raise ValueError("read_out_method should be in {'sum', 'mean'}")
        self.dense1 = nn.Linear(in_features=2*hidden_dim, out_features=input_dim)
        self.dense  = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dense2 = nn.Linear(in_features=hidden_dim * self.channels, out_features=output_dim)


    def forward(self, x, adj_matrix, membership=None):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, input_dim) else
        :param padded_neighbors: (node_num, max_degree), int64, each row represents the 1-hop neighbors for node_i;
                                `-1` are padded to indicate invalid values.
        :param adj_matrix: (node_num, node_num)
        :param membership: (batch_size, node_num) with element `1` indicating to which graph a given node belongs
        :return:
        """
        #--- optional embedding ---#
        original_x = x
        if self.num_embedding > 0:
            x = self.emb0.forward(x)

        graph_reps = []
        total_loss_restore = 0
        x0 = x
        membership0 = membership
        for i in range(self.channels):
            #--- aggregation ---#
            top_x, bottom_x, membership = self.GUNs[i](adj_matrix, x0, membership0)
            # top_x = x

            #--- readout ---#
            x = bottom_x
            x = torch.spmm(membership, x)      # sum
            if self.readout_method == 'mean':
                node_num = torch.spmm(membership, torch.ones((membership.shape[1], 1)).to(membership.device))
                x = x / node_num
            x = F.dropout(F.relu(self.dense1(x)), p=self.dropout, training=self.training)
            graph_representation = F.relu(self.dense(x))
            graph_reps.append(graph_representation)

            #--- restore input node during training stage ---#
            if self.training:
                if self.num_embedding > 0:
                    restored_x = self.decode(top_x)
                    loss_restore = torch.nn.CrossEntropyLoss()(restored_x, original_x)
                else:
                    restored_x = top_x
                    loss_restore = torch.nn.MSELoss()(restored_x, original_x)
                total_loss_restore += loss_restore
        x = torch.cat(graph_reps, dim=1)
        x = self.dense2(x)

        if self.training:
            return x, total_loss_restore
        else:
            return x


#--- variant based on model_2
class GConv_v0(nn.Module):
    """
    A basic graph *convolution* module
    The `backbone` is affiliated with the so call *convolution* op, i.e., A*x for further feature processing.
    By default a simple 2-layer dense module is used (with relu activation). Or you can specify a more complex module
    yourself.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 aggregation_method='mean',  # equals to the so called "GCN"
                 backbone=None,
                 eps=None
                 ):
        """
        :param input_dim:
        :param output_dim:
        :param aggregation_method:  {'sum', 'mean', 'max', 'sum_max', 'mean_max'}
        :param backbone: nn.Module for feature transformation, if not given, a two-layer dense module will be used by default.
        :param eps: set to None to disable the eps learning.
        """
        super().__init__()
        self.input_dim      = input_dim
        self.output_dim     = output_dim
        self.aggregation_method = aggregation_method.lower()
        if self.aggregation_method not in {'sum', 'mean', 'max', 'sum_max', 'mean_max'}:
            raise ValueError("aggregation_method should be in {'sum', 'mean', 'max', 'sum_max', 'mean_max'}")
        if self.aggregation_method in {'sum_max', 'mean_max'}:
            self.alpha = Parameter(torch.scalar_tensor(0.1, dtype=torch.float32))        # to be learned
            torch.nn.utils.clip_grad_value_(self.alpha, 0.05)
        if eps is None:
            self.eps = None
        else:
            self.eps        = Parameter(torch.scalar_tensor(eps, dtype=torch.float32))        # to be learned
            torch.nn.utils.clip_grad_value_(self.eps, 0.05)
        if backbone is None:
            hidden_dim = min(2 * input_dim, 256)
            self.backbone = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(in_features=hidden_dim, out_features=output_dim),
                                          nn.ReLU(),
                                          BatchNorm1d(num_features=output_dim))
        else:
            self.backbone = backbone


    def forward(self, x, padded_neighbors=None, adj_matrix=None, self_connection=True):
        """
        :param x: (node_num, D)
        :param padded_neighbors:
        :param adj_matrix: (node_num, node_num), adjacency matrix, sparse or dense, normalized or not
        :param self_connection: whether or not add self-connection
        :return:
        """
        x_org = x

        if self.aggregation_method == 'max':
            x = max_pooling_GIN(x, padded_neighbors)
        else:
            x = torch.spmm(adj_matrix, x)
            if self.aggregation_method == 'mean':
                degree = torch.spmm(adj_matrix, torch.ones((adj_matrix.shape[0], 1)).to(adj_matrix.device))
                x = x / degree
            if self.aggregation_method.endswith('_max'):
                x = x + self.alpha * max_pooling_GIN(x, padded_neighbors)

        if self_connection:
            x = x + x_org
        if self.eps is not None:
            x = x + self.eps * x_org

        x = self.backbone(x)
        return x

class model_4(nn.Module):
    """
    model_4 is based on the structure of model_2
    """
    def __init__(self,
                 num_embedding=0,
                 block_num=4,
                 input_dim=75,
                 hidden_dim=256,
                 output_dim=None,     # class num
                 dropout=0.5,
                 aggregation_method='sum',
                 readout_method='sum',
                 eps=None,                   # set to None to disable learning this parameter
                 add_dense_connection=False  # whether add dense connection among the blocks
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
        self.block_num          = block_num
        self.dropout            = dropout
        self.aggregation_method = aggregation_method
        self.readout_method     = readout_method
        if self.readout_method not in {'sum', 'mean'}:
            raise ValueError("read_out_method should be in {'sum', 'mean'}")
        self.add_dense_connection = add_dense_connection
        self.blocks  = nn.ModuleList()
        for i in range(self.block_num):
            self.blocks.append(GConv_v0(input_dim=input_dim, output_dim=input_dim,
                                        aggregation_method=aggregation_method,
                                        eps=eps))
        self.dense0 = nn.Linear(in_features=input_dim*(self.block_num+1), out_features=hidden_dim)
        self.dense1 = nn.Linear(in_features=hidden_dim, out_features=input_dim)
        self.bn0    = BatchNorm1d(num_features=input_dim)
        self.dense2 = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x, padded_neighbors=None, adj_matrix=None, membership=None):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param padded_neighbors: (node_num, max_degree), int64, each row represents the 1-hop neighbors for node_i;
                                `-1` are padded to indicate invalid values.
        :param adj_matrix: (node_num, node_num), adjacency matrix, sparse or dense
        :param membership: (batch_size, node_num) with element `1` indicating to which graph a given node belongs, sparse or dense
        :return: (batch_size, class_num)
        """
        if self.num_embedding > 0:
            x = self.emb0.forward(x)
        #--- aggregation ---#
        hiddens = [x]
        block_input = x
        for i in range(self.block_num):
            x = self.blocks[i](x=block_input, padded_neighbors=padded_neighbors, adj_matrix=adj_matrix, self_connection=True)
            if self.add_dense_connection:
                block_input = block_input + x
            else:
                block_input = x
            x = F.dropout(x, p=self.dropout, training=self.training)  # todo: alternative: dropout before/after the block input
            hiddens.append(x)
        #--- readout ---#
        graph_representations = []
        for i in range(self.block_num+1):
            pooled = torch.spmm(membership, hiddens[i])
            if self.readout_method == 'mean':
                node_num = torch.spmm(membership, torch.ones((membership.shape[1], 1)).to(membership.device))
                pooled = pooled / node_num
            graph_representations.append(pooled)
        x = torch.cat(graph_representations, dim=1)

        x = self.dense0(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dense1(x)
        x = self.bn0(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dense2(x)
        return x

class GConv_v1(nn.Module):
    """
    A generic graph *convolution* module. With different arg combinations, this generic module will be equivalent
    to various published graph *convolution* operations.
    1) [GCN convolution](https://arxiv.org/abs/1609.02907): aggregation_methods=['sum'], self_connection=False,
       adjacency_matrix is normalized and with self connection
    2) [GIN convolution](https://arxiv.org/abs/1810.00826): aggregation_methods=['sum'], eps=0, self_connection=True,
       adjacency_matrix is un-normalized and without self connection
    3) [GraphSAGE convolution](https://arxiv.org/abs/1706.02216): aggregation_methods=['mean'], self_connection=False,
       adjacency_matrix is un-normalized and with self connection
    4) [GraphConv convolution](https://arxiv.org/abs/1810.02244): aggregation_methods=['sum'], self_connection=True,
       affine_before_merge=True, adjacency_matrix is un-normalized and without self connection
    5) [Gated convoluiton](https://arxiv.org/abs/1511.05493): aggregation_methods=['rnn'], self_connection=True,
       adjacency_matrix is un-normalized and without self connection
    The `backbone` is affiliated with the so call *convolution* op for further feature processing.
    By default a simple 2-layer dense module is used (with relu activation). Or you can specify a more complex module
    yourself.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 aggregation_methods=('sum', 'max'),
                 multiple_aggregation_merge='cat',
                 affine_before_merge=False,
                 backbone=None,
                 eps=None,
                 ):
        """
        :param input_dim:
        :param output_dim:
        :param aggregation_methods: tuple of strings in {'sum', 'mean', 'max'}
        :param multiple_aggregation_merge: {'sum', 'cat'}, if there are multiple aggregation methods, how their results are merged
        :param affine_before_merge: if True, the output of each neighborhood aggregation method will be further affine-transformed
        :param backbone: nn.Module for feature transformation, if not given, a two-layer dense module will be used by default.
        :param eps: set to None to disable the eps learning.
        """
        super().__init__()
        self.input_dim           = input_dim
        self.output_dim          = output_dim
        self.affine_before_merge = affine_before_merge
        self.aggregation_methods = []
        for item in aggregation_methods:
            item = item.lower()
            if item not in {'sum', 'mean', 'max', 'rnn'}:
                raise ValueError("aggregation_method should be in {'sum', 'mean', 'max', 'rnn'}")
            self.aggregation_methods.append(item)
            if item == 'rnn':
                self.affine_before_rnn = nn.Linear(in_features=input_dim, out_features=input_dim)
                self.rnn = nn.GRUCell(input_size=input_dim, hidden_size=input_dim)
        self.multiple_aggregation_merge = multiple_aggregation_merge.lower()
        assert self.multiple_aggregation_merge in {'sum', 'cat'}
        aggregation_num = len(self.aggregation_methods)
        if self.affine_before_merge:
            self.affine_tranforms = nn.ModuleList()
            for i in range(aggregation_num):
                self.affine_tranforms.append(nn.Linear(in_features=input_dim, out_features=input_dim))
        self.alpha = [1.0]        # default dummy value
        if aggregation_num > 1:
            if self.multiple_aggregation_merge == 'sum':
                self.alpha = Parameter(torch.tensor(np.ones(aggregation_num), dtype=torch.float32))  # to be learned
                torch.nn.utils.clip_grad_value_(self.alpha, 0.1)
            else:
                self.merge_layer = nn.Linear(in_features=input_dim * aggregation_num, out_features=input_dim)
        if eps is None:
            self.eps = None
        else:
            self.eps = Parameter(torch.scalar_tensor(eps, dtype=torch.float32))        # to be learned
            torch.nn.utils.clip_grad_value_(self.eps, 0.1)
        if backbone is None:
            hidden_dim = min(2 * input_dim, 256)
            self.backbone = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(in_features=hidden_dim, out_features=output_dim),
                                          nn.ReLU(),
                                          BatchNorm1d(num_features=output_dim))
        else:
            self.backbone = backbone


    def forward(self, x, padded_neighbors=None, adj_matrix=None, self_connection=True):
        """
        :param x: (node_num, D)
        :param padded_neighbors:
        :param adj_matrix: (node_num, node_num), adjacency matrix, sparse or dense, normalized or not
        :param self_connection: whether or not add self-connection
        :return:
        """
        x_org = x
        aggr_outputs = []
        for i, aggr_method in enumerate(self.aggregation_methods):
            if aggr_method == 'max':
                x = neighbor_op(x_org, padded_neighbors, op='max')
            elif aggr_method == 'rnn':
                x = self.affine_before_rnn(x_org)
                x = torch.spmm(adj_matrix, x)
                x = self.rnn(x, x_org)
            else:
                x = torch.spmm(adj_matrix, x_org)
                if aggr_method == 'mean':
                    degree = torch.spmm(adj_matrix, torch.ones((adj_matrix.shape[0], 1)).to(adj_matrix.device))
                    x = x / degree
            if self.affine_before_merge:
                x = self.affine_tranforms[i](x)
            aggr_outputs.append(x)
        if self.multiple_aggregation_merge == 'sum':
            x = 0
            for i, aggr_out in enumerate(aggr_outputs):
                x += self.alpha[i] * aggr_out
        else:
            if len(self.aggregation_methods) > 1:
                x = torch.cat(aggr_outputs, dim=1)
                x = self.merge_layer(x)
            else:
                x = aggr_outputs[0]

        if self_connection:
            x = x + x_org
        if self.eps is not None:
            x = x + self.eps * x_org

        x = self.backbone(x)
        return x

class model_4v1(nn.Module):
    """
    model_4 is based on the structure of model_2
    """
    def __init__(self,
                 num_embedding=0,
                 block_num=5,
                 input_dim=75,
                 hidden_dim=256,
                 output_dim=None,     # class num
                 aggregation_methods=('sum', 'max'),
                 affine_before_merge=False,
                 multiple_aggregation_merge='cat',
                 readout_method='sum',
                 eps=0.1,                   # set to None to disable learning this parameter
                 add_dense_connection=True  # whether add dense connection among the blocks
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
        self.block_num           = block_num
        self.aggregation_methods = aggregation_methods
        self.multiple_aggregation_merge = multiple_aggregation_merge
        self.readout_method      = readout_method
        if self.readout_method not in {'sum', 'mean'}:
            raise ValueError("read_out_method should be in {'sum', 'mean'}")
        self.add_dense_connection = add_dense_connection
        self.blocks  = nn.ModuleList()
        for i in range(self.block_num):
            self.blocks.append(GConv_v1(input_dim=input_dim, output_dim=input_dim,
                                        aggregation_methods=aggregation_methods,
                                        multiple_aggregation_merge=multiple_aggregation_merge,
                                        affine_before_merge=affine_before_merge,
                                        eps=eps))
        self.dense0 = nn.Linear(in_features=input_dim*(self.block_num+1), out_features=hidden_dim)
        self.dense1 = nn.Linear(in_features=hidden_dim, out_features=input_dim)
        self.bn0    = BatchNorm1d(num_features=input_dim)
        self.dense2 = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x, padded_neighbors=None, adj_matrix=None, membership=None, dropout=0.0):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param padded_neighbors: (node_num, max_degree), int64, each row represents the 1-hop neighbors for node_i;
                                `-1` are padded to indicate invalid values.
        :param adj_matrix: (node_num, node_num), adjacency matrix, sparse or dense
        :param membership: (batch_size, node_num) with element `1` indicating to which graph a given node belongs, sparse or dense
        :param dropout: dropout value
        :return: (batch_size, class_num)
        """
        if self.num_embedding > 0:
            x = self.emb0.forward(x)
        #--- aggregation ---#
        hiddens = [x]
        block_input = x
        for i in range(self.block_num):
            block_input = F.dropout(block_input, p=dropout, training=self.training)
            x = self.blocks[i](x=block_input, padded_neighbors=padded_neighbors, adj_matrix=adj_matrix, self_connection=True)
            if self.add_dense_connection:
                block_input = block_input + x
            else:
                block_input = x
            # x = F.dropout(x, p=self.dropout, training=self.training)  # todo: alternative: dropout before/after the block input
            hiddens.append(x)
        #--- readout ---#
        graph_representations = []
        for i in range(self.block_num+1):
            pooled = torch.spmm(membership, hiddens[i])
            if self.readout_method == 'mean':
                node_num = torch.spmm(membership, torch.ones((membership.shape[1], 1)).to(membership.device))
                pooled = pooled / node_num
            graph_representations.append(pooled)
        x = torch.cat(graph_representations, dim=1)

        x = self.dense0(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense1(x)
        x = self.bn0(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense2(x)
        return x

'''
1. model_4v2 exhibits high performance variance in spite of its large model size
2. the thought behind `GConv_v2` is that the reason A*x is called *convolution* in GNN is because graph has no
   *ordered* neighborhood structure, so only operations treat each neighbor node equally can be used. A*x is equivalent
   to summation of all neighbor nodes with equal weighting. With this limitation, there's no much choice for designing 
   neighbor nodes operations. So, if we can affiliate some kind *order* to the neighborhood, then we can use ops which
   don't have to treat each neighbor node equally.
3. Sadly, the above thought doesn't come through.
'''
class GConv_v2(nn.Module):
    """
    A neighborhood convolution module.
    The neighbors are presumed sorted by certain rules.
    """
    def __init__(self,
                 conv_in_channels=None,
                 conv_out_channels=None,
                 kernel_size=3,
                 merge_method='max',
                 ):

        super().__init__()
        assert merge_method in {'cat', 'sum', 'mean', 'max'}
        self.merge_method = merge_method
        self.conv0 = nn.Conv1d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=kernel_size)

    def forward(self, x, padded_neighbors=None, self_node=True):
        """
        :param x: (node_num, D)
        :param padded_neighbors: (node_num, max_degree_in_batch)
        :param self_node: whether include self nodes in convolution
        :return: x: (node_num, D_out), D_out = conv_out_channels if `merge_method` in {'sum', 'mean', 'max'}, else = conv_out_channels * T_out
                    in which `T_out` = max_degree_in_batch + 1 - kernel_size
        """
        node_num = x.shape[0]
        if self_node:
            self_nodes = torch.arange(0, node_num, dtype=padded_neighbors.dtype, device=padded_neighbors.device)
            padded_neighbors = torch.cat([self_nodes.reshape((-1, 1)), padded_neighbors], dim=1)   # (node_num, max_degree + 1)

        pad_value = torch.mean(x, dim=0)                         # (D,)
        padded_x  = torch.cat([x, pad_value.reshape((1, -1))])   # (node_num+1, D)
        x = padded_x[padded_neighbors]    # (node_num, max_degree+1, D)
        x = x.transpose(1, 2)             # (node_num, D, T_in=max_degree+1)
        x = self.conv0(x)                 # (node_num, conv_out_channels, T_out)  #todo: non-linear activation?

        if self.merge_method == 'sum':
            x = torch.sum(x, dim=2)
        elif self.merge_method == 'mean':
            x = torch.mean(x, dim=2)
        elif self.merge_method == 'max':
            x = torch.max(x, dim=2)[0]
        else:
            x = torch.reshape(x, (node_num, -1))
        return x

class model_4v2(nn.Module):
    """
    Use GConv_v2 as basic block, unsuccessful.
    """
    def __init__(self,
                 num_embedding=0,
                 block_num=5,
                 input_dim=75,
                 hidden_dim=512,
                 output_dim=None,     # class num
                 readout_method='sum',
                 add_dense_connection=True  # whether add dense connection among the blocks
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
        self.block_num           = block_num
        self.readout_method      = readout_method
        if self.readout_method not in {'sum', 'mean'}:
            raise ValueError("read_out_method should be in {'sum', 'mean'}")
        self.add_dense_connection = add_dense_connection
        self.blocks  = nn.ModuleList()
        for i in range(self.block_num):
            self.blocks.append(GConv_v2(conv_in_channels=hidden_dim,
                                        conv_out_channels=hidden_dim,
                                        kernel_size=3,
                                        merge_method='max'))
        self.dense0 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dense1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dense2 = nn.Linear(in_features=hidden_dim, out_features=input_dim)
        self.bn0    = BatchNorm1d(num_features=input_dim)
        self.dense3 = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x, padded_neighbors=None, membership=None, dropout=0.0, **kwargs):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param padded_neighbors: (node_num, max_degree), int64, each row represents the 1-hop neighbors for node_i;
                                `-1` are padded to indicate invalid values.
        :param adj_matrix: (node_num, node_num), adjacency matrix, sparse or dense
        :param membership: (batch_size, node_num) with element `1` indicating to which graph a given node belongs, sparse or dense
        :param dropout: dropout value
        :return: (batch_size, class_num)
        """
        if self.num_embedding > 0:
            x = self.emb0(x)
        x = self.dense0(x)
        #--- aggregation ---#
        hiddens = [x]
        block_input = x
        for i in range(self.block_num):
            block_input = F.dropout(block_input, p=dropout, training=self.training)
            x = self.blocks[i](x=block_input, padded_neighbors=padded_neighbors, self_node=True)
            if self.add_dense_connection:
                block_input = block_input + x
            else:
                block_input = x
            hiddens.append(x)
        #--- readout ---#
        graph_representations = 0
        for i in range(self.block_num+1):
            pooled = torch.spmm(membership, hiddens[i])
            if self.readout_method == 'mean':
                node_num = torch.spmm(membership, torch.ones((membership.shape[1], 1)).to(membership.device))
                pooled = pooled / node_num
            graph_representations += pooled
        x = graph_representations
        x = self.dense1(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense2(x)
        x = self.bn0(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense3(x)
        return x


