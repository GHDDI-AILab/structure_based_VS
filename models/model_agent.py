from functools import partial

from models import data_loader
from models import model_sbvs
from models.model_graph import model_4v4
from models.model_classifier import cls_1

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_Agent():
    """
    model agent idea from LBVS project
    """
    def __init__(self,
                 cfg,
                 device=-1,
                 atom_dict=None,
                 ):
        super().__init__()

        self.model_ver  = cfg['MODEL_VER']['model_ver']

        if isinstance(device, torch.device):
            self.device = device
        elif device < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' %device)
        
        batch_data_loader = getattr(data_loader, cfg['MODEL_VER']['batch_loader'])
        if atom_dict is not None:
            self.batch_data_loader = partial(batch_data_loader, atom_dict=atom_dict)
        model = getattr(model_sbvs, self.model_ver)
        self.model = model(cfg, num_embedding=len(atom_dict))
        self.model = self.model.to(device)

    def forward(self, batch_data, dropout=0, **kwargs):
        # NonDockingGG
        if self.model_ver.startswith('model_ndgg'):
            data_ligand, data_protein, *aux = batch_data
            for i in range(len(data_ligand)):
                data_ligand[i] = torch.from_numpy(data_ligand[i]).to(self.device)
            for i in range(len(data_protein)):
                data_protein[i] = torch.from_numpy(data_protein[i]).to(self.device)
            scorematrix = self.model((data_ligand, data_protein),
                                    dropout=dropout, *aux)
        
        elif self.model_ver.startswith('model_ndsg'):
            data_ligand, data_protein, *aux = batch_data
            for i in range(len(data_ligand)):
                data_ligand[i] = torch.from_numpy(data_ligand[i]).to(self.device)
            for i in range(len(data_protein)):
                data_protein[i] = torch.from_numpy(data_protein[i]).to(self.device)
            scorematrix = self.model((data_ligand, data_protein),
                                    dropout=dropout, *aux)

        return scorematrix

