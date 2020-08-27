import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_ext.module import BatchNorm1d
import torch_scatter
import numpy as np

from torch_scatter import scatter_max, scatter_min, scatter_mean
from torch_scatter import scatter_add, scatter_softmax


class GraphConv(nn.Module):
    """
    A generic graph *convolution* module.

    Note:
    * when `degree_wise` mode is enabled, inputs of `forward()` are required to be specially formatted.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 aggregation_methods=('sum', 'max'),       # {'sum', 'mean', 'max', 'min', 'att'}
                 multiple_aggregation_merge_method='cat',  # {'cat', 'sum'}
                 affine_before_merge=False,
                 update_method='rnn',                      # {'cat', 'sum', 'rnn', 'max'}
                 backbone='default',
                 degree_wise=False,
                 max_degree=1,
                 **kwargs
                 ):
        """
        :param input_dim:
        :param output_dim:
        :param aggregation_methods: tuple of strings in  {'sum', 'mean', 'max', 'min', 'att'}
        :param multiple_aggregation_merge_method: {'cat', 'sum'}, how their results should be merged
                                                  if there are multiple aggregation methods simultaneously
        :param affine_before_merge: if True, output of each neighborhood aggregation method will be further
                                    affine-transformed before they are merged
        :param update_method: {'cat', 'sum', 'rnn', 'max'}, how the center node feature should be merged with aggregated neighbor feature
        :param backbone: nn.Module for feature transformation, a two-layer dense module will be used by default, you can
                         set it to `None` to disable this transformation.
        :param degree_wise: set it to True to enable degree-wise neighborhood aggregation
        :param max_degree: maximum degree allowed. If you have a few nodes with degree > `max_degree` and you don't want to treat
                           them separately degree-wise, you just need put them into a certain degree group and feed the corresponding
                           `x, edges, degree_slices` accordingly. You can even further "quantitize" the degree groups by treating
                           certain degree values as a single value, for example, say you have `max_degree` = 10, you can make a "coarser"
                           degree-wise operation by treating nodes with 0 degree as a group, nodes with 1, 2, 3, 4, 5 degrees as a group,
                           and nodes with 6, 7, 8, 9, 10 & > 10 degrees as a group.
        :param kwargs:  1) head_num: attention head number, for `att` aggregation method, default = 1
                        2) att_mode: {'combo', 'single'}, specify attention mode for `att` aggregation method. The `att`
                           method is basically correlating node features with the attention vector, this correlation can
                           be done at single node level or at neighbor-center combination level. For the latter mode, attention
                           is done on concatenation of each tuple of (neighbor, center) node features.
        """
        super().__init__()
        self.input_dim           = input_dim
        self.output_dim          = output_dim
        self.affine_before_merge = affine_before_merge
        self.aggregation_methods = []
        for item in aggregation_methods:
            item = item.lower()
            if item not in {'sum', 'mean', 'max', 'min', 'att'}:
                raise ValueError("aggregation_method should be in {'sum', 'mean', 'max', 'min', 'att'}")
            self.aggregation_methods.append(item)
            if item == 'att':
                if 'head_num' in kwargs:
                    self.head_num = kwargs['head_num']
                else:
                    self.head_num = 1
                assert self.input_dim % self.head_num == 0, 'input_dim must be multiple of head_num'
                if 'att_mode' in kwargs:
                    self.att_mode = kwargs['att_mode']
                else:
                    self.att_mode = 'combo'
                assert self.att_mode in {'single', 'combo'}
                if self.att_mode == 'single':
                    self.att_weight = Parameter(torch.empty(size=(1, self.head_num, self.input_dim//self.head_num)))
                else:
                    self.att_weight = Parameter(torch.empty(size=(1, self.head_num, 2 * self.input_dim // self.head_num)))
        self.multiple_aggregation_merge_method = multiple_aggregation_merge_method.lower()
        assert self.multiple_aggregation_merge_method in {'cat', 'sum'}
        aggregation_num = len(self.aggregation_methods)
        if self.affine_before_merge:
            self.affine_transforms = nn.ModuleList()
            for i in range(aggregation_num):
                self.affine_transforms.append(nn.Linear(in_features=input_dim, out_features=input_dim))
        if aggregation_num > 1:
            if self.multiple_aggregation_merge_method == 'sum':
                pass
            else:
                self.merge_layer = nn.Linear(in_features=input_dim * aggregation_num, out_features=input_dim)
        self.update_method = update_method.lower()
        assert self.update_method in {'cat', 'sum', 'rnn', 'max'}
        if self.update_method == 'rnn':
            self.rnn = nn.GRUCell(input_size=input_dim, hidden_size=input_dim)

        if backbone is not None and backbone.lower() == 'default':
            if self.update_method == 'cat':
                backbone_input_dim = 2 * input_dim
            else:
                backbone_input_dim = input_dim
            self.backbone = nn.Sequential(nn.Linear(in_features=backbone_input_dim, out_features=256),
                                          nn.LeakyReLU(),
                                          BatchNorm1d(num_features=256),
                                          nn.Linear(in_features=256, out_features=output_dim),
                                          nn.LeakyReLU(),
                                          BatchNorm1d(num_features=output_dim),
                                          )
        else:
            self.backbone = backbone

        self.degree_wise = degree_wise
        if self.degree_wise:
            self.max_degree      = max_degree
            self.linear_neighbor = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
            self.linear_center   = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
            self.linears_degree  = nn.ModuleList()
            for i in range(self.max_degree):
                self.linears_degree.append(nn.Linear(in_features=input_dim, out_features=input_dim, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'att_weight'):
            nn.init.xavier_normal_(self.att_weight)

    def forward(self, x, edges, edge_weights=None, include_self_in_neighbor=False, degree_slices=None):
        # type: (Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tensor
        """
        Forward with degree-wise aggregation support
        :param x: (node_num, D), requiring that nodes of the same degree are grouped together when self.degree_wise = True
        :param edges: (2, edge_num), requiring that edges with center node of the same degree are grouped together when
                      self.degree_wise = True. Note due to this requirement `include_self_in_neighbor` argument is not
                      supported when self.degree_wise = True, you need to add self connections in `edges` before feeding
                      the data into the module.
        :param edge_weights: (edge_num,), edge weights, order the same with `edges`
        :param include_self_in_neighbor: when performing neighborhood operations, whether include self (center) nodes, note
                                         this argument must be set to False when self.degree_wise = True, in that case, you
                                         need to add self connections in `edges` before feeding the data into the module
        :param degree_slices: (max_degree_in_batch+1, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                              to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                              required when self.degree_wise = True, otherwise leave it to None
        :return:
        """
        node_num, feature_dim = x.shape
        x_org = x
        if include_self_in_neighbor:
            if self.degree_wise:
                raise ValueError('`include_self_in_neighbor` param must be set to False when `degree_wise`=True')
            else:
                edges, edge_weights = add_remaining_self_loops(edges, edge_weights, num_nodes=node_num)
        x_neighbor  = x_org[edges[0, :], :]
        if self.degree_wise:
            x_neighbor  = self.linear_neighbor(x_neighbor)
            x           = self.linear_center(x)

        if degree_slices is None:
            edge_num = edges.shape[1]
            degree_slices = np.array([[0, edge_num]], dtype=np.int64)
            degree_slices = torch.from_numpy(degree_slices).to(x.device)

        x_aggregated_degree_list = []
        for degree, span in enumerate(degree_slices):
            if self.degree_wise:
                node_num_degree = (span[1] - span[0]) // max(degree, 1)
                # node_num_degree = np.int64(node_num_degree)
            else:
                node_num_degree = node_num
            if node_num_degree <= 0:
                continue
            if self.degree_wise and degree == 0:  # no neighbors
                x_aggregated_degree = torch.zeros(node_num_degree, feature_dim, dtype=x.dtype, device=x.device)
                x_aggregated_degree_list.append(x_aggregated_degree)
                continue
            edges_degree        = edges[:, span[0]:span[1]]   # no copy
            edge_weights_degree = edge_weights[span[0]:span[1]] if edge_weights is not None else None
            x_neighbor_degree   = x_org[edges_degree[0, :], :]
            if edge_weights_degree is not None:
                x_neighbor_degree = x_neighbor_degree * edge_weights_degree
            if self.degree_wise:
                x_neighbor_degree = self.linears_degree[degree](x_neighbor_degree)
                x_neighbor_degree += x_neighbor[span[0]:span[1], :]

            #---- neighborhood aggregation ----#
            aggr_outputs = []
            if self.degree_wise:
                scatter_index = edges_degree[1, :] - edges_degree[1, :].min()
            else:
                scatter_index = edges_degree[1, :]
            for i, aggr_method in enumerate(self.aggregation_methods):
                if aggr_method == 'max':
                    x_aggregated_degree, _ = scatter_max(x_neighbor_degree, scatter_index, dim=0, dim_size=node_num_degree)
                elif aggr_method == 'min':
                    x_aggregated_degree, _ = scatter_min(x_neighbor_degree, scatter_index, dim=0, dim_size=node_num_degree)
                elif aggr_method == 'mean':
                    x_aggregated_degree = scatter_mean(x_neighbor_degree, scatter_index, dim=0, dim_size=node_num_degree)
                elif aggr_method == 'sum':   # aggr_method == 'sum'
                    x_aggregated_degree = scatter_add(x_neighbor_degree, scatter_index, dim=0, dim_size=node_num_degree)
                elif aggr_method == 'att':
                    edge_num_degree = span[1] - span[0]
                    query = x_neighbor_degree.view(edge_num_degree, self.head_num, -1)  # (N, D) -> (N, heads, out_channels)
                    if self.att_mode == 'combo':
                        x_center = x[edges_degree[1, :], :].view(edge_num_degree, self.head_num, -1)
                        query = torch.cat([query, x_center], dim=-1)  # (N, heads, 2*out_channels)
                    alpha = query * self.att_weight
                    alpha = alpha.sum(dim=-1)         # (N, heads)
                    alpha = F.leaky_relu(alpha, 0.2)  # (N, heads), use leaky relu as in GAT paper
                    alpha = scatter_softmax(alpha, scatter_index.view(edge_num_degree, 1), dim=0)
                    x_neighbor_degree = x_neighbor_degree.view(edge_num_degree, self.head_num, -1) * alpha.view(-1, self.head_num, 1)  # (N, heads, out_channels)
                    x_neighbor_degree = x_neighbor_degree.view(edge_num_degree, -1)
                    x_aggregated_degree = scatter_add(x_neighbor_degree, scatter_index, dim=0, dim_size=node_num_degree)
                else:
                    raise ValueError('aggregation method = %s not supported' % aggr_method)
                if self.affine_before_merge:
                    x_aggregated_degree = self.affine_transforms[i](x_aggregated_degree)
                aggr_outputs.append(x_aggregated_degree)
            if self.multiple_aggregation_merge_method == 'sum':
                x_aggregated_degree = 0
                for i, aggr_out in enumerate(aggr_outputs):
                    x_aggregated_degree += aggr_out
            else:  # concatenation
                if len(self.aggregation_methods) > 1:
                    x_aggregated_degree = torch.cat(aggr_outputs, dim=1)
                    x_aggregated_degree = self.merge_layer(x_aggregated_degree)  # for dimension normalization
                else:
                    x_aggregated_degree = aggr_outputs[0]
            x_aggregated_degree_list.append(x_aggregated_degree)

        x_aggregated = torch.cat(x_aggregated_degree_list, dim=0)

        #---- center update ---#
        if self.update_method == 'sum':
            x += x_aggregated
        elif self.update_method == 'cat':
            x = torch.cat([x, x_aggregated], dim=1)
        elif self.update_method == 'rnn':
            x = self.rnn(x, x_aggregated)
        elif self.update_method == 'max':
            x = torch.max(torch.stack([x, x_aggregated]), dim=0)[0]
        else:
            raise ValueError('update method = %s not supported' % self.update_method)

        if self.backbone is not None:
            x = self.backbone(x)
        return x

class GraphReadout(nn.Module):
    """
    A generic graph readout op, supported method = {'sum', 'mean', 'max', 'min', 'att', 'rnn-<op1>-<op2>'}, in which
    <op1> & <op2> can be any among {'sum', 'mean', 'max', 'min'}
    :param degree_wise: set it to True to enable degree-wise graph readout
    :param max_degree: maximum degree allowed, required when `degree_wise` = True
    :param kwargs:  1) att_mode: {'single', 'combo'}, when op = 'att', specify attention mode. Default = 'single', The `att`
                       method is basically correlating node features with the attention vector, this correlation can
                       be done at single node level or at neighbor-center combination level. For the latter mode, attention
                       is done on concatenation of each tuple of (neighbor, center) node features. The center here is a
                       pseudo center constructed by summing all the node features in each graph.
                    2) head_num: default = 1, attention head number
    """
    def __init__(self,
                 input_dim=None,
                 readout_methods=('max', 'sum'),
                 multiple_readout_merge_method='cat',   # {'cat', 'sum'}
                 affine_before_merge=False,
                 degree_wise=False,
                 max_degree=1,
                 **kwargs,
                 ):
        super().__init__()
        self.input_dim           = input_dim
        self.readout_methods     = readout_methods
        self.multiple_readout_merge_method = multiple_readout_merge_method.lower()
        self.affine_before_merge = affine_before_merge
        self.rnns                = nn.ModuleList()
        self.att_weights         = nn.ParameterList()
        self.affine_transforms   = nn.ModuleList()
        self.degree_wise         = degree_wise
        self.max_degree          = max_degree

        for readout_method in readout_methods:
            readout_method = readout_method.lower()
            if readout_method not in {'sum', 'mean', 'max', 'min', 'att'} and not readout_method.startswith('rnn'):
                raise ValueError("readout_method should be in {'sum', 'mean', 'max', 'min', 'att' or 'rnn-<op1>-<op2>'} but got %s" % readout_method)

            if readout_method.startswith('rnn'):
                assert input_dim is not None, 'input_dim must be specified for `rnn-...` read out method'
                self.rnns.append(nn.GRUCell(input_size=input_dim, hidden_size=input_dim))
            elif readout_method == 'att':
                assert input_dim is not None, 'input_dim must be specified for `att` read out method'
                if 'head_num' in kwargs:
                    self.head_num = kwargs['head_num']
                else:
                    self.head_num = 1
                assert input_dim % self.head_num == 0, 'input_dim must be multiple of head_num'
                if 'att_mode' in kwargs:
                    self.att_mode = kwargs['att_mode']
                else:
                    self.att_mode = 'single'
                assert self.att_mode in {'single', 'combo'}
                if self.att_mode == 'single':
                    self.att_weights.append(Parameter(torch.empty(size=(1, self.head_num, input_dim // self.head_num))))
                else:
                    self.att_weights.append(Parameter(torch.empty(size=(1, self.head_num, 2 * input_dim // self.head_num))))

        if self.degree_wise:
            feature_dim = input_dim * (self.max_degree + 1)
        else:
            feature_dim = input_dim
        readout_method_num = len(self.readout_methods)
        if readout_method_num > 1:
            if self.multiple_readout_merge_method == 'cat':
                self.merge_layer = nn.Linear(in_features=feature_dim * readout_method_num, out_features=feature_dim)
        if self.affine_before_merge:
            for i in range(readout_method_num):
                self.affine_transforms.append(nn.Linear(in_features=feature_dim, out_features=feature_dim))


        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'att_weights'):
            for w in self.att_weights:
                nn.init.xavier_normal_(w)

    def forward(self, x, membership, degree_slices=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
        """
        :param x: (node_num, feature_dim)
        :param membership: (node_num,), int64, representing to which graph the i_th node belongs
        :param degree_slices: (max_degree_in_batch+1, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                              to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                              required when self.degree_wise = True, otherwise leave it to None
        :return x_readout: (B, feature_dim) if self.degree_wise = False, else (B, feature_dim * (self.max_degree + 1))
        """
        node_num, feature_dim = x.shape
        B = torch.max(membership) + 1           # batch size

        if degree_slices is None:
            degree_slices = np.array([[0, node_num]], dtype=np.int64)
            degree_slices = torch.from_numpy(degree_slices).to(x.device)

        x_readout_list    = []
        rnns              = iter(self.rnns)
        att_weights       = iter(self.att_weights)
        affine_transforms = iter(self.affine_transforms)
        for readout_method in self.readout_methods:
            x_readout_degree_list = []
            start_idx = 0
            if readout_method == 'att':
                att_weight = next(att_weights)
            elif readout_method.startswith('rnn'):
                rnn = next(rnns)
            for degree, span in enumerate(degree_slices):
                if self.degree_wise:
                    node_num_degree = (span[1] - span[0]) // max(degree, 1)
                else:
                    node_num_degree = node_num

                end_idx = start_idx + node_num_degree
                if node_num_degree <= 0:
                    x_readout_degree = torch.zeros(B, feature_dim, dtype=x.dtype, device=x.device)
                    x_readout_degree_list.append(x_readout_degree)
                    continue
                else:
                    x_degree = x[start_idx: end_idx, :]
                    membership_degree = membership[start_idx: end_idx]
                    start_idx = end_idx
                if readout_method.startswith('rnn'):
                    op_center, op_neighbor = readout_method.split('-')[1:]
                    h = F.gelu(x_degree)
                    pseudo_nodes = []
                    for op in [op_center, op_neighbor]:
                        if op == 'mean':
                            pseudo_node    = scatter_mean(h, membership_degree, dim=0, dim_size=B)
                        elif op == 'max':
                            pseudo_node, _ = scatter_max(h, membership_degree, dim=0, dim_size=B)
                        elif op == 'min':
                            pseudo_node, _ = scatter_min(h, membership_degree, dim=0, dim_size=B)
                        else:  # op_neighbor ='sum'
                            pseudo_node    = scatter_add(h, membership_degree, dim=0, dim_size=B)
                        pseudo_nodes.append(pseudo_node)
                    x_readout_degree = rnn(pseudo_nodes[0], pseudo_nodes[1])
                elif readout_method == 'att':
                    N = x_degree.shape[0]
                    x_neighbor = x_degree.view(N, self.head_num, -1)  # (N, D) -> (N, heads, out_channels)
                    query = x_neighbor
                    if self.att_mode == 'combo':
                        x_center = scatter_add(x_degree, membership_degree, dim=0, dim_size=B)[membership_degree, :]
                        x_center = x_center.view(N, self.head_num, -1)
                        query = torch.cat([query, x_center], dim=-1)  # (N, heads, 2*out_channels)
                    alpha = query * att_weight
                    alpha = alpha.sum(dim=-1)  # (N, heads)
                    alpha = F.leaky_relu(alpha, 0.2)  # (N, heads), use leaky relu as in GAT paper
                    alpha = scatter_softmax(alpha, membership_degree.view(N, 1), dim=0)
                    x_neighbor = x_neighbor * alpha.view(-1, self.head_num, 1)  # (N, heads, out_channels)
                    x_neighbor = x_neighbor.view(N, -1)
                    x_readout_degree = scatter_add(x_neighbor, membership_degree, dim=0,  dim_size=B)
                elif readout_method == 'mean':
                    x_readout_degree = scatter_mean(x_degree, membership_degree, dim=0, dim_size=B)
                elif readout_method == 'max':
                    x_readout_degree, _ = scatter_max(x_degree, membership_degree, dim=0, dim_size=B)
                elif readout_method == 'min':
                    x_readout_degree, _ = scatter_min(x_degree, membership_degree, dim=0, dim_size=B)
                else:  # 'sum'
                    x_readout_degree = scatter_add(x_degree, membership_degree, dim=0, dim_size=B)
                x_readout_degree_list.append(x_readout_degree)

            if self.degree_wise:
                for i in range(degree+1, self.max_degree+1):
                    x_readout_degree = torch.zeros(B, feature_dim, dtype=x.dtype, device=x.device)
                    x_readout_degree_list.append(x_readout_degree)

            x_readout = torch.cat(x_readout_degree_list, dim=1)  # (B, feature_dim * (max_degree+1))
            if self.affine_before_merge:
                x_readout = next(affine_transforms)(x_readout)
            x_readout_list.append(x_readout)

        if self.multiple_readout_merge_method == 'sum':
            x_readout = 0
            for i, out in enumerate(x_readout_list):
                x_readout += out
        else:  # concatenation
            if len(self.readout_methods) > 1:
                x_readout = torch.cat(x_readout_list, dim=1)
                x_readout = self.merge_layer(x_readout)  # for dimension normalization
            else:
                x_readout = x_readout_list[0]

        return x_readout

class GraphConv_v3(nn.Module):
    """
    A generic graph *convolution* module.

    This module provides two different version of `forward()` functions. When you have torch-scatter package available under
    you environment, `forward_scatter()` is preferred, otherwise use `forward_neighbor_op()` since the latter doesn't require
    3rd party package installation.

    Note:
    * `forward_scatter()` and `forward_neighbor_op()` require different formatted inputs. It's not difficult to convert
    data format between each other, just keep in mind the format difference.
    * `att` aggregation method is only implemented in `forward_scatter()` branch, if there is no significant demand, there's
    NO schedule for implementing this method in `forward_neighbor_op()` branch.

    The `backbone` is affiliated with the so call *convolution* op for further feature processing.
    By default a simple 2-layer dense module is used (with relu activation). You can also specify a more complex module
    yourself.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 aggregation_methods=('sum', 'rnn-max'),
                 multiple_aggregation_merge='cat',
                 affine_before_merge=False,
                 backbone='default',
                 eps=1.0,
                 use_neighbor_op=False,
                 **kwargs
                 ):
        """
        :param input_dim:
        :param output_dim:
        :param aggregation_methods: tuple of strings in
                                    {'sum', 'mean', 'max', 'min',
                                     'rnn-sum', 'rnn-mean', 'rnn-max', 'rnn-min',
                                     'att'}
        :param multiple_aggregation_merge: {'sum', 'cat'}, how their results should be merged
                                           if there are multiple aggregation methods simultaneously
        :param affine_before_merge: if True, the output of each neighborhood aggregation method will be further
                                    affine-transformed
        :param backbone: nn.Module for feature transformation, a two-layer dense module will be used by default, you can
                         set it to `None` to disable this transformation.
        :param eps: initial value for the self-connection weight, learnable. Set to `None` to disable this additional
                    self-connection as well as the associated `eps` learning.
        :param use_neighbor_op: whether use `neighbor_op` for graph node neighborhood operations. If you have `torch-scatter`
                                available, set this to `False` to enable the `torch-scatter` speedup.
        :param kwargs:  1) head_num: attention head number, for `att` aggregation method
                        2) att_mode: {'combo', 'single'}, specify attention mode for `att` aggregation method. The `att`
                           method is basically correlating node features with the attention vector, this correlation can
                           be done at single node level or at neighbor-center combination level. For the latter mode, attention
                           is done on concatenation of each tuple of (neighbor, center) node features.
        """
        super().__init__()
        self.input_dim           = input_dim
        self.output_dim          = output_dim
        self.affine_before_merge = affine_before_merge
        self.aggregation_methods = []
        for item in aggregation_methods:
            item = item.lower()
            if item not in {'sum', 'mean', 'max', 'min', 'rnn-sum', 'rnn-mean', 'rnn-max', 'rnn-min', 'att'}:
                raise ValueError("aggregation_method should be in {'sum', 'mean', 'max', 'min', 'rnn-sum', 'rnn-mean', 'rnn-max', 'rnn-min', 'att'}")
            self.aggregation_methods.append(item)
            if item.startswith('rnn'):
                self.affine_before_rnn = nn.Linear(in_features=input_dim, out_features=input_dim)
                self.rnn = nn.GRUCell(input_size=input_dim, hidden_size=input_dim)
            elif item == 'att':
                if 'head_num' in kwargs:
                    self.head_num = kwargs['head_num']
                else:
                    self.head_num = 1
                assert self.input_dim % self.head_num == 0, 'input_dim must be multiple of head_num'
                if 'att_mode' in kwargs:
                    self.att_mode = kwargs['att_mode']
                else:
                    self.att_mode = 'combo'
                assert self.att_mode in {'single', 'combo'}
                if self.att_mode == 'single':
                    self.att_weight = Parameter(torch.empty(size=(1, self.head_num, self.input_dim//self.head_num)))
                else:
                    self.att_weight = Parameter(torch.empty(size=(1, self.head_num, 2 * self.input_dim // self.head_num)))
        self.multiple_aggregation_merge = multiple_aggregation_merge.lower()
        assert self.multiple_aggregation_merge in {'sum', 'cat'}
        aggregation_num = len(self.aggregation_methods)
        if self.affine_before_merge:
            self.affine_transforms = nn.ModuleList()
            for i in range(aggregation_num):
                self.affine_transforms.append(nn.Linear(in_features=input_dim, out_features=input_dim))
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
        if backbone.lower() == 'default':
            hidden_dim = min(2 * input_dim, 256)
            self.backbone = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(in_features=hidden_dim, out_features=output_dim),
                                          nn.ReLU(),
                                          BatchNorm1d(num_features=output_dim))
        else:
            self.backbone = backbone
        self.use_neighbor_op = use_neighbor_op
        if self.use_neighbor_op:
            self.forward = self.forward_neighbor_op
        else:
            self.forward = self.forward_scatter
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'att_weight'):
            nn.init.xavier_normal_(self.att_weight)

    def forward_neighbor_op(self, x, padded_neighbors=None, padded_edge_weights=None,
                            include_self_in_neighbor=False, self_connection=False):
        """
        Forward with neighbor_* ops
        :param x: (node_num, D)
        :param padded_neighbors: as required by `neighbor_op`.
        :param padded_edge_weights: as required by `neighbor_op`. You can leave it alone if no edge weighting/normalization
                                    is necessary.
        :param include_self_in_neighbor: when performing neighborhood operations, whether include self nodes
        :param self_connection: whether or not add self-connection in the final output. This parameter is for compliance
                                with published papers. You can safely ignore this parameter and use the `self.eps` only.
        :return:
        """
        x_org = x
        aggr_outputs = []
        for i, aggr_method in enumerate(self.aggregation_methods):
            if aggr_method.startswith('rnn'):
                x = self.affine_before_rnn(x_org)
                op = aggr_method.split('-')[1]
                x = neighbor_op(x, padded_neighbors, op=op, include_self=include_self_in_neighbor, padded_edge_weights=padded_edge_weights)
                x = self.rnn(x, x_org)
            else:
                x = neighbor_op(x_org, padded_neighbors, op=aggr_method, include_self=include_self_in_neighbor, padded_edge_weights=padded_edge_weights)
            if self.affine_before_merge:
                x = self.affine_transforms[i](x)
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
            x = x + self.eps * x_org   # from GIN

        if self.backbone is not None:
            x = self.backbone(x)
        return x


    def forward_scatter(self, x, edges, edge_weights=None, include_self_in_neighbor=False, self_connection=False):
        """
        Forward with scatter_* ops
        :param x: (node_num, D)
        :param edges: (2, edge_num)
        :param edge_weights: (edge_num,), edge weights
        :param include_self_in_neighbor: when performing neighborhood operations, whether include self nodes
        :param self_connection: whether or not add self-connection in the final output. This parameter is for compliance
                                with published papers. You can safely ignore this parameter and use the `self.eps` only.
        :return:
        """
        x_org = x
        node_num = x.shape[0]
        edge_num = edges.shape[1]
        if include_self_in_neighbor:
            edges, edge_weights = add_remaining_self_loops(edges, edge_weights, num_nodes=node_num)
        aggr_outputs = []
        for i, aggr_method in enumerate(self.aggregation_methods):
            if aggr_method.startswith('rnn'):
                x = self.affine_before_rnn(x_org)
                x_neighbor = x[edges[0, :], :]
                if edge_weights is not None:
                    x_neighbor = x_neighbor * edge_weights
                op = aggr_method.split('-')[1]
                if op == 'max':
                    x, _ = torch_scatter.scatter_max(x_neighbor, edges[1,:], dim=0, dim_size=node_num)
                elif op == 'min':
                    x, _ = torch_scatter.scatter_min(x_neighbor, edges[1,:], dim=0, dim_size=node_num)
                elif op == 'mean':
                    x = torch_scatter.scatter_mean(x_neighbor, edges[1, :], dim=0, dim_size=node_num)
                else:  # op = 'sum'
                    x = torch_scatter.scatter_add(x_neighbor, edges[1, :], dim=0, dim_size=node_num)
                x = self.rnn(x, x_org)
            elif aggr_method == 'att':
                x_neighbor = x_org[edges[0, :], :].view(edge_num, self.head_num, -1)  # (N, D) -> (N, heads, out_channels)
                x = x_neighbor
                if self.att_mode == 'combo':
                    x_center = x_org[edges[1, :], :].view(edge_num, self.head_num, -1)
                    x = torch.cat([x, x_center], dim=-1) # (N, heads, 2*out_channels)
                alpha = x * self.att_weight
                alpha = alpha.sum(dim=-1)         # (N, heads)
                alpha = F.leaky_relu(alpha, 0.2)  # (N, heads), use leaky relu as in GAT paper
                alpha = torch_scatter.composite.scatter_softmax(alpha, edges[1, :].view(edge_num,1), dim=0)
                x_neighbor = x_neighbor * alpha.view(-1, self.head_num, 1)  # (N, heads, out_channels)
                x_neighbor = x_neighbor.view(edge_num, -1)
                x = torch_scatter.scatter_add(x_neighbor, edges[1, :], dim=0, dim_size=node_num)
            elif aggr_method == 'max':
                x_neighbor = x_org[edges[0, :], :]
                if edge_weights is not None:
                    x_neighbor = x_neighbor * edge_weights
                x, _ = torch_scatter.scatter_max(x_neighbor, edges[1, :], dim=0, dim_size=node_num)
            elif aggr_method == 'min':
                x_neighbor = x_org[edges[0, :], :]
                if edge_weights is not None:
                    x_neighbor = x_neighbor * edge_weights
                x, _ = torch_scatter.scatter_min(x_neighbor, edges[1, :], dim=0, dim_size=node_num)
            elif aggr_method == 'mean':
                x_neighbor = x_org[edges[0, :], :]
                if edge_weights is not None:
                    x_neighbor = x_neighbor * edge_weights
                x = torch_scatter.scatter_mean(x_neighbor, edges[1, :], dim=0, dim_size=node_num)
            else:   # aggr_method == 'sum'
                x_neighbor = x_org[edges[0,:], :]
                if edge_weights is not None:
                    x_neighbor = x_neighbor * edge_weights
                x = torch_scatter.scatter_add(x_neighbor, edges[1,:], dim=0, dim_size=node_num)
            if self.affine_before_merge:
                x = self.affine_transforms[i](x)
            aggr_outputs.append(x)
        if self.multiple_aggregation_merge == 'sum':
            x = 0
            for i, aggr_out in enumerate(aggr_outputs):
                x += self.alpha[i] * aggr_out
        else:  # concatenation
            if len(self.aggregation_methods) > 1:
                x = torch.cat(aggr_outputs, dim=1)
                x = self.merge_layer(x)  # for dimension normalization
            else:
                x = aggr_outputs[0]

        if self_connection:
            x = x + x_org
        if self.eps is not None:
            x = x + self.eps * x_org   # from GIN

        if self.backbone is not None:
            x = self.backbone(x)
        return x

class model_4v3(nn.Module):
    """
    use pseudo center node & aggregation op at readout stage
    """
    def __init__(self,
                 num_embedding=0,
                 block_num=5,
                 input_dim=75,
                 hidden_dim=256,
                 output_dim=None,     # class num
                 aggregation_methods=('rnn-max', 'sum'),
                 affine_before_merge=False,
                 multiple_aggregation_merge='cat',
                 readout_method='rnn-sum-max',
                 eps=1.0,                   # set to None to disable learning this parameter
                 add_dense_connection=True,  # whether add dense connection among the blocks
                 use_neighbor_op=False,
                 **kwargs
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
        self.block_num           = block_num
        self.aggregation_methods = aggregation_methods
        self.multiple_aggregation_merge = multiple_aggregation_merge
        self.readout_method      = readout_method
        for substr in self.readout_method.split('-'):
            if substr not in {'sum', 'mean', 'max', 'min', 'rnn', 'att'}:
                raise ValueError("invalid readout method specification")
        self.add_dense_connection = add_dense_connection
        self.blocks  = nn.ModuleList()
        for i in range(self.block_num):
            self.blocks.append(GraphConv_v3(input_dim=input_dim, output_dim=input_dim,
                                            aggregation_methods=aggregation_methods,
                                            multiple_aggregation_merge=multiple_aggregation_merge,
                                            affine_before_merge=affine_before_merge,
                                            eps=eps,
                                            use_neighbor_op=use_neighbor_op))
        if readout_method.startswith('rnn'):
            self.affine_before_rnns = nn.ModuleList()
            self.rnns = nn.ModuleList()
            for i in range(self.block_num+1):
                self.affine_before_rnns.append(nn.Linear(in_features=input_dim, out_features=input_dim))
                self.rnns.append(nn.GRUCell(input_size=input_dim, hidden_size=input_dim))
        elif readout_method == 'att':
            if 'head_num' in kwargs:
                self.head_num = kwargs['head_num']
            else:
                self.head_num = 1
            assert input_dim % self.head_num == 0, 'input_dim must be multiple of head_num'
            if 'att_mode' in kwargs:
                self.att_mode = kwargs['att_mode']
            else:
                self.att_mode = 'combo'
            assert self.att_mode in {'single', 'combo'}
            self.att_weights = nn.ParameterList()
            for i in range(self.block_num + 1):
                if self.att_mode == 'single':
                    att_weight = Parameter(torch.empty(size=(1, self.head_num, input_dim // self.head_num)))
                else:
                    att_weight = Parameter(torch.empty(size=(1, self.head_num, 2 * input_dim // self.head_num)))
                self.att_weights.append(att_weight)

        self.dense0 = nn.Linear(in_features=input_dim*(self.block_num+1), out_features=hidden_dim)
        self.dense1 = nn.Linear(in_features=hidden_dim, out_features=input_dim)
        self.bn0    = BatchNorm1d(num_features=input_dim)
        self.dense2 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.use_neighbor_op = use_neighbor_op
        if self.use_neighbor_op:
            self.forward = self.forward_neighbor_op
        else:
            self.forward = self.forward_scatter
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'att_weights'):
            for att_weight in self.att_weights:
                nn.init.xavier_normal_(att_weight)

    def forward_neighbor_op(self, x, padded_neighbors=None, padded_membership=None, dropout=0.0):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param padded_neighbors: (node_num, max_degree), int64, each row represents the 1-hop neighbors for node_i;
                                `-1` are padded to indicate invalid values.
        :param membership: (batch_size, node_num) with element `1` indicating to which graph a given node belongs,
                           sparse or dense
        :param dropout: dropout value
        :return: x, (batch_size, class_num)
        """
        if self.num_embedding > 0:
            x = self.emb0.forward(x)
        #--- aggregation ---#
        hiddens = [x]
        block_input = x
        for i in range(self.block_num):
            block_input = F.dropout(block_input, p=dropout, training=self.training)
            x = self.blocks[i](x=block_input, padded_neighbors=padded_neighbors,
                               include_self_in_neighbor=False,
                               self_connection=False)
            if self.add_dense_connection:
                block_input = block_input + x
            else:
                block_input = x
            # x = F.dropout(x, p=self.dropout, training=self.training)  # todo: alternative: dropout before/after the block input
            hiddens.append(x)
        #--- readout ---#
        graph_representations = []
        for i in range(self.block_num+1):
            # pooled = torch.spmm(membership, hiddens[i])
            if self.readout_method.startswith('rnn'):
                op_center, op_neighbor = self.readout_method.split('-')[1:]
                h = self.affine_before_rnns[i](hiddens[i])
                h = F.gelu(h)
                pseudo_neighbor = neighbor_op(h, padded_membership, op=op_neighbor)
                pseudo_center   = neighbor_op(hiddens[i], padded_membership, op=op_center)
                pooled = self.rnns[i](pseudo_neighbor, pseudo_center)
            else:
                pooled = neighbor_op(hiddens[i], padded_membership, op=self.readout_method)
            graph_representations.append(pooled)
        x = torch.cat(graph_representations, dim=1)

        x = self.dense0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense1(x)
        x = self.bn0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense2(x)
        return x

    def forward_scatter(self, x, edges=None, membership=None, dropout=0.0):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
        :param membership: (node_num,), int64, representing to which graph the i_th node belongs
        :param dropout: dropout value
        :return: x, (batch_size, class_num)
        """
        if self.num_embedding > 0:
            x = self.emb0.forward(x)
        #--- aggregation ---#
        hiddens = [x]
        block_input = x
        for i in range(self.block_num):
            block_input = F.dropout(block_input, p=dropout, training=self.training)
            x = self.blocks[i](x=block_input, edges=edges, include_self_in_neighbor=False, self_connection=False)
            if self.add_dense_connection:
                block_input = block_input + x
            else:
                block_input = x
            # x = F.dropout(x, p=self.dropout, training=self.training)  # todo: alternative: dropout before/after the block input
            hiddens.append(x)
        #--- readout ---#
        graph_representations = []
        for i in range(self.block_num+1):
            if self.readout_method.startswith('rnn'):
                op_center, op_neighbor = self.readout_method.split('-')[1:]
                h = self.affine_before_rnns[i](hiddens[i])
                h = F.gelu(h)

                if op_neighbor == 'mean':
                    pseudo_neighbor = torch_scatter.scatter_mean(h, membership, dim=0)
                elif op_neighbor == 'max':
                    pseudo_neighbor, _ = torch_scatter.scatter_max(h, membership, dim=0)
                elif op_neighbor == 'min':
                    pseudo_neighbor, _ = torch_scatter.scatter_min(h, membership, dim=0)
                else:     # op_neighbor ='sum'
                    pseudo_neighbor = torch_scatter.scatter_add(h, membership, dim=0)

                if op_center == 'mean':
                    pseudo_center = torch_scatter.scatter_mean(hiddens[i], membership, dim=0)
                elif op_center == 'max':
                    pseudo_center, _ = torch_scatter.scatter_max(hiddens[i], membership, dim=0)
                elif op_center == 'min':
                    pseudo_center, _ = torch_scatter.scatter_min(hiddens[i], membership, dim=0)
                else:     # 'sum'
                    pseudo_center = torch_scatter.scatter_add(hiddens[i], membership, dim=0)

                pooled = self.rnns[i](pseudo_neighbor, pseudo_center)
            elif self.readout_method == 'att':
                N = hiddens[i].shape[0]
                x_neighbor = hiddens[i].view(N, self.head_num, -1)  # (N, D) -> (N, heads, out_channels)
                x = x_neighbor
                if self.att_mode == 'combo':
                    x_center = torch_scatter.scatter_add(hiddens[i], membership, dim=0)[membership,:]
                    x_center = x_center.view(N, self.head_num, -1)
                    x = torch.cat([x, x_center], dim=-1) # (N, heads, 2*out_channels)
                alpha = x * self.att_weights[i]
                alpha = alpha.sum(dim=-1)         # (N, heads)
                alpha = F.leaky_relu(alpha, 0.2)  # (N, heads), use leaky relu as in GAT paper
                alpha = torch_scatter.composite.scatter_softmax(alpha, membership.view(N,1), dim=0)
                x_neighbor = x_neighbor * alpha.view(-1, self.head_num, 1)  # (N, heads, out_channels)
                x_neighbor = x_neighbor.view(N, -1)
                pooled = torch_scatter.scatter_add(x_neighbor, membership, dim=0)
            else:
                if self.readout_method == 'mean':
                    pooled = torch_scatter.scatter_mean(hiddens[i], membership, dim=0)
                elif self.readout_method == 'max':
                    pooled, _ = torch_scatter.scatter_max(hiddens[i], membership, dim=0)
                elif self.readout_method == 'min':
                    pooled, _ = torch_scatter.scatter_min(hiddens[i], membership, dim=0)
                else:     # 'sum'
                    pooled = torch_scatter.scatter_add(hiddens[i], membership, dim=0)

            graph_representations.append(pooled)
        x = torch.cat(graph_representations, dim=1)

        #--- classification ---#
        x = self.dense0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense1(x)
        x = self.bn0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense2(x)
        return x

class model_4v4(nn.Module):
    """
    Degree-wise convolution as well as degree-wise readout
    """
    def __init__(self,
                 num_embedding=0,
                 block_num=5,
                 input_dim=75,
                 hidden_dim=256,
                 output_dim=2,     # class num
                 degree_wise=False,
                 max_degree=1,
                 aggregation_methods=('max', 'sum'),
                 multiple_aggregation_merge_method='sum',
                 affine_before_merge=False,
                 node_feature_update_method='rnn',
                 readout_methods=('rnn-sum-max',),
                 multiple_readout_merge_method='sum',
                 add_dense_connection=True,  # whether add dense connection among the blocks
                 pyramid_feature=True,
                 slim=True,
                 **kwargs
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
        self.block_num                  = block_num
        self.degree_wise                = degree_wise
        self.max_degree                 = max_degree
        self.aggregation_methods        = aggregation_methods
        self.multiple_aggregation_merge = multiple_aggregation_merge_method
        self.readout_methods            = readout_methods
        self.add_dense_connection       = add_dense_connection
        self.pyramid_feature            = pyramid_feature
        self.slim                       = slim
        self.blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.blocks.append(GraphConv(input_dim=input_dim, output_dim=input_dim,
                                         aggregation_methods=aggregation_methods,
                                         multiple_aggregation_merge_method=multiple_aggregation_merge_method,
                                         affine_before_merge=affine_before_merge,
                                         update_method=node_feature_update_method,
                                         degree_wise=degree_wise,
                                         max_degree=max_degree,
                                         backbone='default',
                                         **kwargs,
                                         ))
        self.readout_ops = nn.ModuleList()
        if self.pyramid_feature:
            readout_block_num = self.block_num + 1
        else:
            readout_block_num = 1
        self.readout_block_num = readout_block_num
        for i in range(readout_block_num):
            self.readout_ops.append(GraphReadout(readout_methods=self.readout_methods, input_dim=input_dim,
                                                 multiple_readout_merge_method=multiple_readout_merge_method,
                                                 affine_before_merge=affine_before_merge,
                                                 degree_wise=degree_wise,
                                                 max_degree=max_degree,
                                                 **kwargs))
            if self.slim:
                break
        readout_dim = input_dim * readout_block_num
        if self.degree_wise:
            readout_dim *= (self.max_degree + 1)
        self.dense0 = nn.Linear(in_features=readout_dim, out_features=hidden_dim)
        self.bn0    = BatchNorm1d(num_features=hidden_dim)
        self.dense1 = nn.Linear(in_features=hidden_dim, out_features=input_dim)
        self.bn1    = BatchNorm1d(num_features=input_dim)
        self.dense2 = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x, edges=None, membership=None, dropout=0.0, degree_slices=None):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
        :param membership: (node_num,), int64, representing to which graph the i_th node belongs
        :param dropout: dropout value
        :param degree_slices: (max_degree_in_batch, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                              to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                              required when self.degree_wise = True, otherwise leave it to None
        :return: x, (batch_size, class_num)
        """
        if self.num_embedding > 0:
            x = self.emb0(x)
        #--- aggregation ---#
        if self.pyramid_feature:
            hiddens = [x]
        block_input = x
        for i in range(self.block_num):
            block_input = F.dropout(block_input, p=dropout, training=self.training)
            x = self.blocks[i](x=block_input, edges=edges,
                               include_self_in_neighbor=False,
                               degree_slices=degree_slices)
            if self.add_dense_connection:
                block_input = block_input + x
            else:
                block_input = x
            if self.pyramid_feature:
                hiddens.append(x)
        #--- readout ---#
        if self.pyramid_feature:
            graph_representations = []
            for i in range(self.block_num+1):
                idx = 0 if self.slim else i
                pooled = self.readout_ops[idx](hiddens[i], membership, degree_slices=degree_slices)
                graph_representations.append(pooled)
            x = torch.cat(graph_representations, dim=1)
        else:
            x = self.readout_ops[0](x, membership, degree_slices=degree_slices)

        #--- classification ---#
        x = self.dense0(x)
        x = self.bn0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.dense2(x)
        return x
