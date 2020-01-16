import os

from dandelion.util import gpickle
from tqdm import tqdm


if __name__ == '__main__':
    path = '/data01/jfwu/code/data/DeepVS_v1/trueinactive_pdbqt/'
    save_path = '/data01/jfwu/code/data/SBVS/trueinactive/pdb_preprocess/atom_dict/'
    proteins = os.listdir(path)
    atoms = []
    aa = []
    for prt in tqdm(proteins):
        prt_path = os.path.join(path, prt)
        pocket_fn = os.path.join(prt_path, prt+'_pocket.pdb')
        with open(pocket_fn, 'r') as fp:
            lns = fp.readlines()
        
        for ln in lns:
            if not ln.startswith('ATOM'):
                continue
            atom = ln[66:].strip()
            if not atom in atoms:
                atoms.append(atom)
            aa.append(atom)
    atom_dict = {}
    for i, atom in enumerate(atoms):
        atom_dict[atom] = i
    gpickle.dump(atom_dict, os.path.join(save_path, 'pocket_atom_dict.gpkl'))