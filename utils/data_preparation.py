# coding: utf-8
"""
Author: Junfeng Wu (junfeng.wu@ghddi.org)
"""
import os
import os.path as osp
from dandelion.util import gpickle


def preparation(arg):
    # create result save folder
    save_path = './%s' % arg.save_root_folder
    if not osp.exists(save_path):
        os.mkdir(save_path)

    # load dataset
    dataset = arg.dataset
    dataset_path = osp.join(arg.dataset_path, dataset)

    atom_dict = gpickle.load(osp.join(dataset_path, 'atom_dict.gpkl'))
    with open(osp.join(dataset_path, 'TrueInactive.list'), 'r') as fp:
        lns = fp.readlines()
    # lns[0] is title: 'protein'
    proteins = [ln.strip() for ln in lns[1:]]
    # load all data
    Xs = []
    Ys = []
    adj_matrix_is = []
    for protein in proteins:
        protein_fn = '%s/processed_data/%s_cm4_nm6_X_adji.gpkl' % (dataset, protein)
        protein_fn = osp.join(arg.dataset_path, protein_fn)
        Xs_tmp, adj_matrix_is_tmp, Y_tmp = gpickle.load(protein_fn)
        Xs.append(Xs_tmp)
        adj_matrix_is.append(adj_matrix_is_tmp)
        Ys.append(Y_tmp)

    data_for_training = (Xs, Ys, adj_matrix_is)
    return atom_dict, proteins, data_for_training
