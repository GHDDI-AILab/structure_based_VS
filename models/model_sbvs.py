import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_graph
from models import model_sequence
from models import model_classifier

class model_ndgg_1(nn.Module):
    """
    NonDockingGG ver2: 4v3 + 4v3 + cat_MLP
    """
    def __init__(self,
                 cfg,
                 num_embedding,
                 ):
        super(model_ndgg_1, self).__init__()
        model_ligand  = getattr(model_graph, cfg['MODEL_VER']['ligand'])
        model_protein = getattr(model_graph, cfg['MODEL_VER']['protein'])
        model_cls     = getattr(model_classifier, cfg['MODEL_VER']['cls'])

        self.emb0 = nn.Embedding(num_embeddings=num_embedding,
                                 embedding_dim=cfg['MODEL_CONFIG']['atom_embedding_dim'])
        self.model_ligand  = model_ligand(num_embedding=-1,
                                      **cfg['MODEL_CONFIG']['config_ligand'])
        self.model_protein = model_protein(num_embedding=-1,
                                      **cfg['MODEL_CONFIG']['config_protein'])
        
        #classifier
        out_dim_ligand  = cfg['MODEL_CONFIG']['config_ligand']['output_dim']
        out_dim_protein = cfg['MODEL_CONFIG']['config_protein']['output_dim']
        input_dim_cls = out_dim_ligand + out_dim_protein
        self.classifier = model_cls(input_dim = input_dim_cls,
                        **cfg['MODEL_CONFIG']['config_cls'])
    
    def forward(self, data, dropout=0.0, aux=None):
        data_ligand, data_protein = data
        X_ligand, edges_ligand, membership_ligand = data_ligand
        X_protein, edges_protein, membership_protein = data_protein
        # sharing embedding layer
        X_ligand = self.emb0(X_ligand)
        X_protein = self.emb0(X_protein)

        # get ligand and graph representation
        output_ligand = self.model_ligand(X_ligand,
                            edges=edges_ligand,
                            membership=membership_ligand,
                            dropout=dropout)
        output_protein = self.model_protein(X_protein,
                            edges=edges_protein,
                            membership=membership_protein,
                            dropout=dropout)
        
        # classifier
        x = torch.cat([output_ligand, output_protein], dim=-1)
        x = self.classifier(x)
        
        return x

class model_ndgg_2(nn.Module):
    """
    NonDockingGG ver2: 4v4 + 4v4 + cat_MLP
    """
    def __init__(self,
                 cfg,
                 num_embedding,
                 ):
        super(model_ndgg_2, self).__init__()
        model_ligand  = getattr(model_graph, cfg['MODEL_VER']['ligand'])
        model_protein = getattr(model_graph, cfg['MODEL_VER']['protein'])
        model_cls     = getattr(model_classifier, cfg['MODEL_VER']['cls'])

        self.emb0 = nn.Embedding(num_embeddings=num_embedding,
                                 embedding_dim=cfg['MODEL_CONFIG']['atom_embedding_dim'])
        self.model_ligand  = model_ligand(num_embedding=-1,
                                      **cfg['MODEL_CONFIG']['config_ligand'])
        self.model_protein = model_protein(num_embedding=-1,
                                      **cfg['MODEL_CONFIG']['config_protein'])
        
        #classifier
        out_dim_ligand  = cfg['MODEL_CONFIG']['config_ligand']['output_dim']
        out_dim_protein = cfg['MODEL_CONFIG']['config_protein']['output_dim']
        input_dim_cls = out_dim_ligand + out_dim_protein
        self.classifier = model_cls(input_dim = input_dim_cls,
                        **cfg['MODEL_CONFIG']['config_cls'])
    
    def forward(self, data, dropout=0.0, degree_slices=None, aux=None):
        data_ligand, data_protein = data
        X_ligand, edges_ligand, membership_ligand = data_ligand
        X_protein, edges_protein, membership_protein = data_protein
        # sharing embedding layer
        X_ligand = self.emb0(X_ligand)
        X_protein = self.emb0(X_protein)

        # get ligand and graph representation
        output_ligand = self.model_ligand(X_ligand,
                            edges=edges_ligand,
                            membership=membership_ligand,
                            dropout=dropout,
                            degree_slices=degree_slices)
        output_protein = self.model_protein(X_protein,
                            edges=edges_protein,
                            membership=membership_protein,
                            dropout=dropout,
                            degree_slices=degree_slices)
        
        # classifier
        x = torch.cat([output_ligand, output_protein], dim=-1)
        x = self.classifier(x)
        
        return x

def rename_state_dict_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('bert.'):
            new_state_dict[k[5:]] = v
        # # remove others: only bert part used
        # else:
        #     new_state_dict[k] = v
    return new_state_dict

class model_ndsg_2(nn.Module):
    """
    NonDockingGG ver2: BERT(embed only) + 4v4 + cat_MLP
    """
    def __init__(self,
                 cfg,
                 num_embedding,
                 ):
        super(model_ndsg_2, self).__init__()
        model_ligand  = getattr(model_graph, cfg['MODEL_VER']['ligand'])
        model_protein = getattr(model_sequence, cfg['MODEL_VER']['protein'])
        model_cls     = getattr(model_classifier, cfg['MODEL_VER']['cls'])

        # self.emb0 = nn.Embedding(num_embeddings=num_embedding,
        #                          embedding_dim=cfg['MODEL_CONFIG']['atom_embedding_dim'])
        self.model_ligand  = model_ligand(num_embedding=num_embedding,
                                      **cfg['MODEL_CONFIG']['config_ligand'])
        bert_config = model_sequence.BertConfig(cfg['MODEL_CONFIG']['config_protein']['bert_config'])
        self.model_protein = model_protein(bert_config)
        # sequence-pretrain
        if cfg['MODEL_CONFIG']['config_protein']['pretrain'] is not None:
            pretrained_file = cfg['MODEL_CONFIG']['config_protein']['pretrain']
            state_dict = torch.load(pretrained_file, map_location=torch.device('cpu'))
            # rename simple because trained in different module name
            state_dict = rename_state_dict_keys(state_dict)
            self.model_protein.load_state_dict(state_dict)

        # classifier
        out_dim_ligand  = cfg['MODEL_CONFIG']['config_ligand']['output_dim']
        out_dim_protein = bert_config.hidden_size
        input_dim_cls = out_dim_ligand + out_dim_protein
        self.classifier = model_cls(input_dim = input_dim_cls,
                        **cfg['MODEL_CONFIG']['config_cls'])
    
    def forward(self, data, dropout=0.0, degree_slices=None, aux=None):
        data_ligand, data_protein = data
        X_ligand, edges_ligand, membership_ligand = data_ligand
        protein_seqs = data_protein[0]
        # get ligand and graph representation
        output_ligand = self.model_ligand(X_ligand,
                            edges=edges_ligand,
                            membership=membership_ligand,
                            dropout=dropout,
                            degree_slices=degree_slices)
        output_protein = self.model_protein(protein_seqs)
        output_protein = output_protein.sum(axis=1)
        
        # classifier
        x = torch.cat([output_ligand, output_protein], dim=-1)
        x = self.classifier(x)
        
        return x
