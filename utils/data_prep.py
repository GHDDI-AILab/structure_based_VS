# coding: utf-8
"""
Author: Junfeng Wu (junfeng.wu@ghddi.org)
"""
import os
import os.path as osp
import numpy as np
import pandas as pd
import msgpack
import msgpack_numpy as m
m.patch()


def load_atom_dict(atom_dict_file):
    if atom_dict_file.endswith('.gpkl'):
        from dandelion.util import gpickle
        atom_dict = gpickle.load(atom_dict_file)
        # in case atom_dict_file contains 3 dicts: atom:idx / idx:atom / atom:count
        if len(atom_dict) == 3:
            atom_dict = atom_dict[0]
    else:
        atom_dict = pd.read_csv(atom_dict_file, keep_default_na=False, index_col = 'atom')
        atom_dict = atom_dict.to_dict()['idx']
    return atom_dict

def load_aadict(aa_dict_file):
    if aa_dict_file.endswith('.gpkl'):
        from dandelion.util import gpickle
        aa_dict = gpickle.load(aa_dict_file)
    else:
        aa_dict = pd.read_csv(aa_dict_file, keep_default_na=False, index_col = 'amino_acids')
        aa_dict = aa_dict.to_dict()['idx']
    return aa_dict

def load_data_0(args):
    return

def load_data_ndgg_1(args, CFG=None):
    atom_dict = load_atom_dict(args.atom_dict_file)

    ligands = pd.read_csv(args.train_ligands)
    smiles = ligands.smiles.values
    label = ligands.label.values
    protein = ligands.protein.values

    with open(args.train_protein_dict, 'rb') as fp:
        proteins_dict = msgpack.unpackb(fp.read(), raw=False)
    
    for k, v in proteins_dict.items():
        proteins_dict[k][0] = np.array([atom_dict[s] if s in atom_dict else atom_dict['UNK']
                                        for s in proteins_dict[k][0]])

    protein_graphs = np.array([proteins_dict[p] for p in protein])
    # data = (unique_id for loo, smiles, proteins_graphs, label)
    return atom_dict, (protein, smiles, protein_graphs, label)

def load_data_ndsg_1(args, CFG):
    aa_dict = load_aadict(args.aa_dict_file)
    atom_dict = load_atom_dict(args.atom_dict_file)

    ligands = pd.read_csv(args.train_ligands)
    smiles = ligands.smiles.values
    label = ligands.label.values
    protein = ligands.protein.values

    with open(args.train_protein_seq_dict, 'rb') as fp:
        proteins_seq_dict = msgpack.unpackb(fp.read(), raw=False)
    
    proteins_seq = {}
    for k, v in proteins_seq_dict.items():
        p_seq = np.array([2] + [aa_dict[s] for s in v] + [3])
        proteins_seq[k] = np.concatenate([p_seq, [aa_dict['<PAD>']] * (CFG.MODEL_CONFIG.config_protein.seq_max_len - len(p_seq))])

    protein_seqs = np.array([proteins_seq[p] for p in protein])
    CFG.aa_dict_len = len(aa_dict)
    # data = (unique_id for loo, smiles, proteins_seqs, label)
    return atom_dict, (protein, smiles, protein_seqs, label)


def load_data_ndsg_1_CL(args, CFG):
    aa_dict = load_aadict(args.aa_dict_file)
    atom_dict = load_atom_dict(args.atom_dict_file)

    ligands = pd.read_csv(args.train_ligands)
    smiles = ligands.smiles.values
    # label = ligands.label.values
    protein = ligands.protein.values

    with open(args.train_protein_seq_dict, 'rb') as fp:
        proteins_seq_dict = msgpack.unpackb(fp.read(), raw=False)

    proteins_seq = {}
    for k, v in proteins_seq_dict.items():
        p_seq = np.array([2] + [aa_dict[s] for s in v] + [3])
        proteins_seq[k] = np.concatenate(
            [p_seq, [aa_dict['<PAD>']] * (CFG.MODEL_CONFIG.config_protein.seq_max_len - len(p_seq))])

    protein_seqs = np.array([proteins_seq[p] for p in proteins_seq.keys()])
    CFG.aa_dict_len = len(aa_dict)
    # data = (unique_id for loo, smiles, proteins_seqs, label)
    protein = np.array(list(proteins_seq.keys()))
    smiles = np.array([smiles[0]] * len(protein)) # dummy
    label = np.ones_like(protein, dtype=np.int)
    return atom_dict, (protein, smiles, protein_seqs, label)
