# coding:utf-8
# Graph processing utility functions, numpy version
# Created   :  10, 28, 2019
# Revised   :  10, 28, 2019
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import sys
import numpy as np
import scipy.sparse as sps
# import networkx as nx

def initialize_edge_dict(edges=None):
    """
    Initialize an edge dict for further processing
    :param edges: list of tuples of (i, j) or equivalent np array. If given, the edge dict will be initialized with the
                  input value. For duplicate edges, an additional index starting from 1 will be appended at the end of
                  the edge key. Normally the edge key will be in the format of 'i,j' (for duplicate edges it will
                  be 'i,j_k')
    :return: an ordered dict initialized emptily or with given `edges`
    """
    if sys.version_info.minor < 6:   # dict is ordered since python 3.6
        from collections import OrderedDict
        edges_new = OrderedDict()    # python doesn't has ordered set yet, use ordered dict instead
    else:
        edges_new = dict()
    if edges is not None:
        for i, j in edges:
            key = '%d,%d' % (i, j)
            if key in edges_new:     # in case duplicate edges
                k = 1
                while '%s_%d' % (key, k) in edges_new:
                    k += 1
                key = '%s_%d' % (key, k)
            edges_new[key] = (i, j)
    return edges_new

def append_symmetric_edges(edges):
    """
    append missing symmetric edges at the end
    :param edges: list of tuples of (i, j) or equivalent np array
    :return edges_new, list of tuples (i, j)
    :return appended_edges, list of tuples (i, j)
    """
    edges_new = initialize_edge_dict(edges)
    appended_edges = []
    for i, j in edges:
        key = '%d,%d' % (j, i)
        if key not in edges_new:
            edges_new[key] = (j, i)
            appended_edges.append((j, i))
    return list(edges_new.values()), appended_edges

def remove_duplicate_edges(edges):
    """
    remove duplicate edges
    :param edges: list of tuples of (i, j) or equivalent np array
    :return: edges_new: list of tuples (i, j)
             index_removed: list of index of edges removed from the input
    """
    edges_new = initialize_edge_dict()
    index_removed = []
    for idx, edge in enumerate(edges):
        i, j = edge
        key = '%d,%d' % (i, j)
        if key not in edges_new:
            edges_new[key] = (i, j)
        else:
            index_removed.append(idx)
    return list(edges_new.values()), index_removed

def remove_symmetric_edges(edges, remove_duplicate=False):
    """
    remove symmetric edges
    :param edges: list of tuples (i, j) or equivalent np array
    :param remove_duplicate: whether meanwhile remove duplicate edges, default to `False`
    :return: edges_new: list of tuples (i, j)
             index_removed: list of index of edges removed from the input
    """
    edges_new = initialize_edge_dict()
    index_removed = []
    for idx, edge in enumerate(edges):
        i, j = edge
        if '%d,%d' % (j, i) not in edges_new:
            key = '%d,%d' % (i, j)
            if key not in edges_new:
                edges_new[key] = (i, j)
            else:                          # duplicate edge
                if not remove_duplicate:
                    k = 1
                    while '%s_%d' % (key, k) in edges_new:
                        k += 1
                    edges_new['%s_%d' % (key, k)] = (i, j)
                else:
                    index_removed.append(idx)
        else:
            index_removed.append(idx)
    return list(edges_new.values()), index_removed

def normalize_dense_adj_matrix(A, eps=1.0):
    """
    Normalize a dense adjacency matrix as `A <- D^(-1/2) * (A+eps*I) * D^(-1/2)`, in which `D` is computed from `A+eps*I`
    For sparse version, check the following `normalize_sparse_adj_matrix()`.
    :param A: adjacency matrix, (node_num, node_num), dim0 = source_nodes, dim1 = target_nodes
    :param eps: additional self-connection strength, i.e., A <- A + eps * I
    :return: A normalized as A <- D^(-1/2) * (A+eps*I) * D^(-1/2)
    """
    node_num = A.shape[0]
    A = A + eps * np.eye(node_num)
    D_in = A.sum(axis=0)     # degree_in, edge weights are all counted in
    # D_out = A.sum(dim=1)    # degree_out, edge weights are all counted in
    mask = D_in == 0
    D_in[mask] = np.inf
    D_inv_sqrt = 1.0/np.sqrt(D_in)
    if isinstance(D_inv_sqrt, np.matrix):
        D_inv_sqrt = np.asarray(D_inv_sqrt).squeeze()  # in case D_inv_sqrt is np.matrix
    D_inv_sqrt = np.diag(D_inv_sqrt)
    A = np.matmul(D_inv_sqrt, A)
    A = np.matmul(A, D_inv_sqrt)
    return A

def normalize_sparse_adj_matrix(adj_i, adj_v, node_num=None, eps=1.0, return_A=False):
    """
    Normalize a sparse adjacency matrix as `A <- D^(-1/2) * (A+eps*I) * D^(-1/2)`, in which `D` is computed from `A+eps*I`
    For dense version, check the above `normalize_dense_adj_matrix()`.
    :param adj_i: ndarray of shape (2, edge_num), each column is an edge in format of (source_node, target_node).
                  Keep in mind that you need to make sure both (i, j) a.w.a. (j, i) are in `adj_i` to keep adjacency matrix
                  symmetric.
    :param adj_v: (edge_num,)  edge weight, usually all 1s
    :param node_num: node number. If not given, it'll be guessed automatically
    :param eps: self-connection strength, i.e., A <- A + eps * I
    :param return_A: bool, if true, the normalized sparse adjacency matrix A (csr format) will be returned; else updated
                     `adj_i` and `adj_v` will be returned, with these two data a pytorch sparse tensor can be constructed
                     conveniently by `torch.sparse.FloatTensor(adj_i, adj_v)`
    :return: as stated above
    """
    if node_num is None:
        node_num = adj_i.max() - adj_i.min() + 1
    A = sps.coo_matrix((adj_v, (adj_i[0,:], adj_i[1,:])), shape=(node_num, node_num)).tocsr()
    A = A + eps * sps.eye(node_num, format='csr')
    D_in = A.sum(axis=0)     # degree_in, edge weights are all counted in
    # D_out = A.sum(dim=1)    # degree_out, edge weights are all counted in
    mask = D_in == 0
    D_in[mask] = np.inf
    D_inv_sqrt = 1.0/np.sqrt(D_in)
    D_inv_sqrt = sps.diags(np.asarray(D_inv_sqrt).squeeze(), 0, format='csr')  # convert np.matrix to np.array
    A = D_inv_sqrt.dot(A)
    A = A.dot(D_inv_sqrt)
    if return_A:
        return A
    else:
        A = A.tocoo()
        adj_i = np.stack([A.row, A.col], axis=0).astype(adj_i.dtype)
        adj_v = A.data.astype(adj_v.dtype)
        return adj_i, adj_v

def calc_degree_for_center_nodes(edges):
    """
    :param edges: list of tuples (neighbor_node, center_node) or equivalent np array
    :return node_degrees, np.array with shape (max_idx+1,)
    """
    if isinstance(edges, list):
        edges = np.array(edges)
    min_idx, max_idx = edges.min(), edges.max()
    node_degrees = np.zeros(shape=(max_idx+1))
    for neighbor_idx, center_idx in edges:
        node_degrees[center_idx] += 1
    return node_degrees
