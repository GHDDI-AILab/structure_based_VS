import numpy as np
import os, psutil, multiprocessing, threading
from rdkit import Chem
# from ..utils.graph_utils import calc_degree_for_center_nodes
import utils.graph_utils

import time

def canonicalize_molecule(molecule, addH=True):
    if addH:
        molecule = Chem.AddHs(molecule)  # add back all hydrogen atoms
    order    = Chem.CanonicalRankAtoms(molecule)  # to get a canonical form here
    molecule = Chem.RenumberAtoms(molecule, order)
    return molecule

def convert_to_degree_wise_format(X, edges, membership):
    """
    Convert loaded data to degree-wise format.
    This function is computation intensive.
    :param X, edges, membership as returned by load_batch_data_4v3/4v4
    :return X, edges, membership, degree_slices as required by model_4v4
    """
    time0 = time.time()
    node_degrees = calc_degree_for_center_nodes(edges.T).astype(np.int64)
    time1 = time.time()
    node_num = len(node_degrees)
    assert node_num == X.shape[0]
    idx_sorted   = np.argsort(node_degrees)
    node_degrees = node_degrees[idx_sorted]   # (node_num,)
    X            = X[idx_sorted]              # (node_num,)
    membership   = membership[idx_sorted]     # (node_num,)
    time2 = time.time()
    idx_map = dict()
    for idx_new, idx_old in enumerate(idx_sorted):
        idx_map[idx_old] = idx_new
    idx_map = [idx_map[key] for key in range(node_num)]
    idx_map = np.array(idx_map).astype(np.int64)
    edges   = idx_map[edges]
    time3 = time.time()
    # todo: codes between time3 and time4 need accelerated
    min_degree, max_degree = node_degrees.min(), node_degrees.max()
    degree_slices = np.zeros((max_degree+1, 2), dtype=np.int64)
    start_idx = 0
    edges_degree_slices = []
    edge_num = edges.shape[1]
    node_idx_range = np.arange(node_num)
    for degree in range(min_degree, max_degree+1):
        mask = node_degrees == degree
        node_num_degree = mask.sum()
        edge_num_degree = node_num_degree * max(degree, 1)

        degree_slices[degree, :] = [start_idx, start_idx + edge_num_degree]
        if degree > 0:
            start_idx += edge_num_degree
            node_idxs_degree = node_idx_range[mask]
            for idx in node_idxs_degree:
                mask_idx = edges[1,:] == idx
                if any(mask_idx):
                    edges_degree_slices.append(edges[:, mask_idx])
    time4 = time.time()
    edges = np.concatenate(edges_degree_slices, axis=1)
    time5 = time.time()
    # print('node degree calc time =', time1-time0)
    # print('sort time =', time2-time1)
    # print('edge rename time =', time3-time2)
    # print('degree_slice time =', time4-time3)
    # print('edge sort time =', time5-time4)
    assert edges.shape[1] == edge_num
    return X, edges, membership, degree_slices

def load_batch_data_ndgg_1(data, batch_sample_idx, atom_dict, add_self_connection=False, degree_wise=False):
    """
    Corresponding to model_4v4, input is SMILES strings
    :param strings: SMILES strings
    :param batch_sample_idx:
    :param atom_dict:
    :param add_self_connection: whether append self-connection in adjacency matrix
    :param degree_wise: whether in degree-wise format
    :param aux_data_list: list of auxiliary data
    :return: X, edges, membership, degree_slices (optional), *aux_data (optional)
             X: (node_num,), index of each node according to `atom_dict`, int64
             edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
             membership: (node_num,), int64, representing to which graph the i_th node belongs
             degree_slices: (max_degree_in_batch+1, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                            to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                            returned only when `degree_wise` = True
             *aux_data: list of auxiliary data organized in one batch
    """
    protein, ligand_smiles, protein_graphs, labels = data

    # ligand part
    batch_size = len(batch_sample_idx)
    tokenized_sequences = []
    edges = []
    start_idxs = [0]
    membership = []
    for i in range(batch_size):
        s = ligand_smiles[batch_sample_idx[i]]
        molecule = Chem.MolFromSmiles(s)
        molecule = canonicalize_molecule(molecule)
        tokenized_seq = []
        for atom in molecule.GetAtoms():
            if atom.GetSymbol() not in atom_dict:
                tokenized_seq.append(atom_dict['UNK'])    # OOV
            else:
                tokenized_seq.append(atom_dict[atom.GetSymbol()])
        tokenized_sequences.extend(tokenized_seq)
        n = len(tokenized_seq)
        membership.extend([i for _ in range(n)])
        start_idxs.append(n + start_idxs[i])
        edge_list = [(b.GetBeginAtomIdx() + start_idxs[i], b.GetEndAtomIdx() + start_idxs[i]) for b in molecule.GetBonds()]
        edge_list_reverse = [(j, i) for (i, j) in edge_list]
        edges.append(edge_list)
        edges.append(edge_list_reverse)  # add symmetric edge
        if add_self_connection:
            edges.append([(j+start_idxs[i],j+start_idxs[i]) for j in range(n)])

    X = np.array(tokenized_sequences, dtype=np.int64)                          # (node_num,), index of each node, int64
    edges = np.concatenate(edges, axis=0).transpose().astype(np.int64)         # (2, n_pair), each column denotes an edge, from node i to node j, int64
    membership = np.array(membership, dtype=np.int64)                          # (node_num,)

    batch_data_ligand = [X, edges, membership]
    if degree_wise:
        X, edges, membership, degree_slices = convert_to_degree_wise_format(X, edges, membership)
        batch_data_ligand.append(degree_slices)
    
    # protein part
    protein_graphs_batch = protein_graphs[batch_sample_idx]
    protein_X = np.concatenate([protein_graphs_batch[i][0] for i in range(batch_size)])
    protein_node_num = [protein_graphs_batch[i][0].shape[0] for i in range(batch_size)]
    protein_node_num_cumsum = np.cumsum([0] + protein_node_num[:-1])
    protein_edges = np.concatenate([protein_graphs_batch[i][1] + protein_node_num_cumsum[i] for i in range(batch_size)], axis=1)
    protein_membership = np.concatenate([np.zeros(protein_node_num[i]) + i for i in range(batch_size)]).astype(np.int64)

    batch_data_protein = [protein_X, protein_edges, protein_membership]

    # combine
    Y = labels[batch_sample_idx]
    batch_data = [(batch_data_ligand, batch_data_protein), Y]
    return batch_data

def load_batch_data_ndsg_1(data, batch_sample_idx, atom_dict, add_self_connection=False, degree_wise=False):
    """
    Corresponding to model_4v4, input is SMILES strings
    :param strings: SMILES strings
    :param batch_sample_idx:
    :param atom_dict:
    :param add_self_connection: whether append self-connection in adjacency matrix
    :param degree_wise: whether in degree-wise format
    :param aux_data_list: list of auxiliary data
    :return: X, edges, membership, degree_slices (optional), *aux_data (optional)
             X: (node_num,), index of each node according to `atom_dict`, int64
             edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
             membership: (node_num,), int64, representing to which graph the i_th node belongs
             degree_slices: (max_degree_in_batch+1, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                            to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                            returned only when `degree_wise` = True
             *aux_data: list of auxiliary data organized in one batch
    """
    protein, ligand_smiles, protein_seqs, labels = data

    # ligand part
    batch_size = len(batch_sample_idx)
    tokenized_sequences = []
    edges = []
    start_idxs = [0]
    membership = []
    for i in range(batch_size):
        s = ligand_smiles[batch_sample_idx[i]]
        molecule = Chem.MolFromSmiles(s)
        molecule = canonicalize_molecule(molecule)
        tokenized_seq = []
        for atom in molecule.GetAtoms():
            if atom.GetSymbol() not in atom_dict:
                tokenized_seq.append(atom_dict['UNK'])    # OOV
            else:
                tokenized_seq.append(atom_dict[atom.GetSymbol()])
        tokenized_sequences.extend(tokenized_seq)
        n = len(tokenized_seq)
        membership.extend([i for _ in range(n)])
        start_idxs.append(n + start_idxs[i])
        edge_list = [(b.GetBeginAtomIdx() + start_idxs[i], b.GetEndAtomIdx() + start_idxs[i]) for b in molecule.GetBonds()]
        edge_list_reverse = [(j, i) for (i, j) in edge_list]
        edges.append(edge_list)
        edges.append(edge_list_reverse)  # add symmetric edge
        if add_self_connection:
            edges.append([(j+start_idxs[i],j+start_idxs[i]) for j in range(n)])

    X = np.array(tokenized_sequences, dtype=np.int64)                          # (node_num,), index of each node, int64
    edges = np.concatenate(edges, axis=0).transpose().astype(np.int64)         # (2, n_pair), each column denotes an edge, from node i to node j, int64
    membership = np.array(membership, dtype=np.int64)                          # (node_num,)

    batch_data_ligand = [X, edges, membership]
    if degree_wise:
        X, edges, membership, degree_slices = convert_to_degree_wise_format(X, edges, membership)
        batch_data_ligand.append(degree_slices)
    
    # protein part
    protein_seqs_batch = protein_seqs[batch_sample_idx]
    
    batch_data_protein = [protein_seqs_batch]

    # combine
    Y = labels[batch_sample_idx]
    batch_data = [(batch_data_ligand, batch_data_protein), Y]
    return batch_data

def feed_sample_batch(batch_data_loader,
                      data,
                      data_queue=None,
                      max_epoch_num=None,
                      batch_size=64, batch_size_min=None,
                      shuffle=False,
                      use_multiprocessing=False,
                      epoch_start_event=None,
                      epoch_done_event=None
                      ):

    me_process = psutil.Process(os.getpid())
    sample_num = data[0].shape[0]
    if batch_size_min is None:
        batch_size_min = batch_size
    batch_size_min = min([batch_size_min, batch_size])
    done, epoch = False, 0
    while not done:
        if use_multiprocessing:
            if me_process.parent() is None:     # parent process is dead
                raise RuntimeError('Parent process is dead, exiting')
        if epoch_start_event is not None:
            epoch_start_event.wait()
            epoch_start_event.clear()
        if epoch_done_event is not None:
            epoch_done_event.clear()
        if shuffle:
            index = np.random.choice(range(sample_num), size=sample_num, replace=False)
        else:
            index = np.arange(sample_num)
        index = list(index)
        end_idx = 0
        while end_idx < sample_num:
            current_batch_size = np.random.randint(batch_size_min, batch_size + 1)
            start_idx = end_idx
            end_idx = min(start_idx + current_batch_size, sample_num)
            batch_sample_idx = index[start_idx:end_idx]
            batch_data = batch_data_loader(data, batch_sample_idx=batch_sample_idx)
            data_queue.put(batch_data)

        if epoch_done_event is not None:
            # time.sleep(3.0)   # most possible jitter time for cross process communication (mp.queue)
            epoch_done_event.set()
        epoch += 1
        if max_epoch_num is not None:
            if epoch >= max_epoch_num:
                done = True

def sync_manager(worker_epoch_start_event_list, worker_epoch_done_event_list, data_queue):
    while 1:
        for event in worker_epoch_done_event_list:
            event.wait()
            event.clear()
        data_queue.put(None)   # tell the queue consumer that the epoch is done
        for event in worker_epoch_start_event_list:
            event.set()

def start_data_loader(batch_data_loader,
                      data_queue,
                      data,
                      batch_size=128, batch_size_min=None,
                      max_epoch_num=None, worker_num=1,
                      use_multiprocessing=False, name=None, shuffle=False):
    if use_multiprocessing:
        Worker = multiprocessing.Process
        Event  = multiprocessing.Event
    else:
        Worker = threading.Thread
        Event  = threading.Event

    sample_num = data[0].shape[0]

    print('create data processes...')
    startidx, endidx, idxstep = 0, 0, sample_num // worker_num
    workers = []
    worker_epoch_start_event_list = []
    worker_epoch_done_event_list  = []
    for i in range(worker_num):
        if max_epoch_num is None:
            worker_epoch_start_event = Event()
            worker_epoch_done_event  = Event()
            worker_epoch_start_event.set()
            worker_epoch_done_event.clear()
            worker_epoch_start_event_list.append(worker_epoch_start_event)
            worker_epoch_done_event_list.append(worker_epoch_done_event)
        else:
            worker_epoch_start_event = None
            worker_epoch_done_event  = None
        startidx = i * idxstep
        if i == worker_num - 1:
            endidx = sample_num
        else:
            endidx = startidx + idxstep
        data_proc = Worker(target=feed_sample_batch,
                           args=(batch_data_loader,
                                 [x[startidx:endidx] for x in data], # data = (smiles, proteins, labels)
                                 data_queue,
                                 max_epoch_num, batch_size, batch_size_min, shuffle,
                                 use_multiprocessing,
                                 worker_epoch_start_event,
                                 worker_epoch_done_event
                                 ),
                           name='%s_thread_%d' % (name, i))
        data_proc.daemon = True
        data_proc.start()
        workers.append(data_proc)

    if max_epoch_num is None:
        sync_manager_proc = Worker(target=sync_manager,
                                   args=(worker_epoch_start_event_list,
                                         worker_epoch_done_event_list,
                                         data_queue),
                                   name='%s_sync_manager' % name)
        sync_manager_proc.daemon = True
        sync_manager_proc.start()
        workers.append(sync_manager_proc)
    return workers


# TESTING CODE
if __name__ == '__main__':
    pass
    # Xs, adj_matrix_is, Y = gpickle.load('prep_task/processed_data/1a1e_cm4_nm6_X_adji.gpkl')

    # data_container = queue.Queue
    # queue_size = 3
    # worker = 1
    # data_queue = data_container(queue_size * worker)
    # t0 = time.time()
    # batch_data_loader = partial(batch_data_loader_type)
    # workers = start_data_loader(batch_data_loader, data_queue, (Xs, Y, adj_matrix_is),
    #                             batch_size=256, batch_size_min=256,
    #                             max_epoch_num=1, worker_num=1,
    #                             use_multiprocessing=False, name='codetest',
    #                             shuffle=False)
    # print(len(data_queue.get()))
    # t1 = time.time()
    # print('done\n')
    # print('dt: %.2f' %(t1-t0))