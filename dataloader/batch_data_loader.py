import numpy as np


def parse_ligs_type(Xs, adj_matrix_is):
    X = np.concatenate(Xs)

    idx_offset = np.cumsum([len(s) for s in Xs])
    idx_offset = np.insert(idx_offset[:-1], 0, 0)

    adj_matrix_i = []
    for i in range(len(adj_matrix_is)):
        adj_matrix_i.append(adj_matrix_is[i] + idx_offset[i])

    adj_matrix_i = np.concatenate(adj_matrix_i, axis=1).astype(np.int64)
    adj_matrix_v = np.ones(adj_matrix_i.shape[1], dtype=np.float32)

    membership_i = np.vstack([[j for j in range(len(Xs)) for i in range(len(Xs[j]))],
                              [i for i in range(len(X))]])
    membership_v = np.ones(membership_i.shape[1], dtype=np.float32)

    padded_neighbors_tmp = [[] for _ in range(len(X))]
    for i in range(len(adj_matrix_i[0])):
        padded_neighbors_tmp[adj_matrix_i[0][i]].append(adj_matrix_i[1][i])
    pad = len(max(padded_neighbors_tmp, key=len))
    padded_neighbors = np.array([i + [-1]*(pad-len(i)) for i in padded_neighbors_tmp])

    return X, adj_matrix_i, adj_matrix_v, membership_i, membership_v, padded_neighbors

def batch_data_loader_type(data, batch_sample_idx):
    """
    :param data: proteins, ligands, pldf
    :param batch_sample_idx: index of selected ligands
    :param atom_dict: atom_dict for embedding
    :return:
    """
    Xs, Y, adj_matrix_is = data
    Xs = [Xs[i] for i in batch_sample_idx]
    Y = Y[batch_sample_idx]
    adj_matrix_is = [adj_matrix_is[i] for i in batch_sample_idx]

    X, adj_matrix_i, adj_matrix_v, membership_i, membership_v, padded_neighbors = parse_ligs_type(Xs, adj_matrix_is)

    return X, Y, membership_i, membership_v, adj_matrix_i, adj_matrix_v, padded_neighbors