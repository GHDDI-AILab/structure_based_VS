import os
import numpy as np
import torch

import dgl

from dandelion.util import gpickle
from tqdm import tqdm


def distance_matrix(pts1, pts2):
    # pts.shape = k, 3
    sumsq1 = (pts1**2).sum(axis=1)
    sumsq2 = (pts2**2).sum(axis=1)
    k1 = pts1.shape[0]
    k2 = pts2.shape[0]
    dis_cor  = sumsq1 + sumsq2.reshape(k2, 1) - 2 * pts2.dot(pts1.T)
    dis_pts1 = sumsq1 + sumsq1.reshape(k1, 1) - 2 * pts1.dot(pts1.T)
    dis_pts2 = sumsq2 + sumsq2.reshape(k2, 1) - 2 * pts2.dot(pts2.T)
    return dis_cor, dis_pts1, dis_pts2


def sig(x, a=0.1):
    return 1/(1+np.exp(-a*x))


def parse_pocket_pdb_distance(lns):
    k = '-1'
    aa_string = []
    ca = []
    for ln in lns:
        if not ln.startswith('ATOM'):
            continue
        # if ln.startswith('HETATM'):
        #     continue
        idx = ln[22:26]
        if ln[11:17].strip() == 'CA':
            xyz = [ln[26:38], ln[38:46], ln[46:54]]
            xyz = [float(s) for s in xyz]
            ca.append(xyz)
        if idx == k:
            continue
        else:
            k = idx
            aa_string.append(k)

    aa_string = [int(s) for s in aa_string]
    ca = np.array(ca)
    dc, _, _ = distance_matrix(ca, ca)
    return aa_string, dc


def parse_pocket_pdb_chain(lns, aa_string, dc, threshold=0.01):
    # dgl graph
    G = dgl.DGLGraph()
    G.add_nodes(len(aa_string))
    G.ndata['idx'] = torch.tensor(aa_string)
    # relative distance
    G.edata['d'] = torch.tensor([])

    # tensor graph
    nodes = torch.stack([torch.arange(len(aa_string)), torch.tensor(aa_string)])
    edges = []
    # search max connect distance
    d = 0
    for i in range(1, len(aa_string)):
        if aa_string[i] - 1 == aa_string[i-1]:
            d = max(d, dc[i][i-1])

    for i in range(len(aa_string)):
        # if there's a connected aa, then dc[i][i-1] and dc[i][i+1] <= d (which is max)
        # acquire a larger d for min(d1, d2, d)
        d1 = d + 1
        d2 = d + 1
        d_tmp = d
        if i > 0:
            d1 = dc[i][i-1]
            d_tmp = min(d_tmp, d1)
        if i < len(aa_string) - 1:
            d2 = dc[i][i+1]
            d_tmp = min(d_tmp, d2)

        # rd: relative distance: 1 - (sig(dc[i][j] - d_tmp) - 0.5) * 2
        for j in range(len(aa_string)):
            if j == i:
                continue
            else:
                rd = 1 - (sig(dc[i][j] - d_tmp) - 0.5) * 2
                if rd > 0.01:
                    # bi-direction
                    G.add_edge(i, j)
                    G.edata['d'][-1] = rd
                    edges.append((i, j, rd))

    return G, (nodes, edges)


def parse_pocket_pdb_aa(lns, aa_string, atom_dict, cutoff=4):
    atom_dict = gpickle.load(atom_dict)
    G = {}
    for ln in lns:
        if not ln.startswith('ATOM'):
            continue
        idx = int(ln[22:26])
        xyz = [ln[26:38], ln[38:46], ln[46:54]]
        xyz = [float(s) for s in xyz]
        atom = [atom_dict[ln[66:].strip()]]
        parse_ln = xyz + atom
        if idx in G:
            G[idx].append(parse_ln)
        else:
            G[idx] = [parse_ln]

    aa_dgl = []
    aa_tensor = []

    for k, v in G.items():
        val = torch.tensor(v)
        nodes = torch.stack([torch.arange(val.shape[0]).float(),
                             torch.tensor(val[:,3])]).int()
        edges = []
        g = dgl.DGLGraph()
        g.add_nodes(val.shape[0])
        g.ndata['atom'] = val[:, 3]
        coord = val[:, :3].numpy()
        dc, _, _, = distance_matrix(coord, coord)
        for i in range(dc.shape[0]):
            for j in range(dc.shape[0]):
                if j==i:
                    continue
                if dc[i, j] < cutoff:
                    g.add_edge(i, j)
                    edges.append((i, j))
        aa_dgl.append(g)
        aa_tensor.append((nodes, edges))

    return aa_dgl, aa_tensor


if __name__ == '__main__':
    path = '/data01/jfwu/code/data/DeepVS_v1/trueinactive_pdbqt/'
    save_path = '/data01/jfwu/code/data/SBVS/trueinactive/pdb_preprocess/processed/'
    atom_dict = '/data01/jfwu/code/data/SBVS/trueinactive/pdb_preprocess/atom_dict/pocket_atom_dict.gpkl'
    proteins = os.listdir(path)
    for prt in tqdm(proteins):
        prt_path = os.path.join(path, prt)
        pocket_fn = os.path.join(prt_path, prt+'_pocket.pdb')
        with open(pocket_fn, 'r') as fp:
            lns = fp.readlines()
        aa_string, dc = parse_pocket_pdb_distance(lns)
        dgl_graph, tensor_graph = parse_pocket_pdb_chain(lns, aa_string, dc)
        dgl_aa, tensor_aa = parse_pocket_pdb_aa(lns, aa_string, atom_dict)

        gpickle.dump(dgl_graph, os.path.join(save_path, prt+'_chain_dgl.gpkl'))
        gpickle.dump(dgl_aa, os.path.join(save_path, prt+'_aa_dgl.gpkl'))
        
        gpickle.dump(tensor_graph, os.path.join(save_path, prt+'_chain_tensor.gpkl'))
        gpickle.dump(tensor_aa, os.path.join(save_path, prt+'_aa_tensor.gpkl'))