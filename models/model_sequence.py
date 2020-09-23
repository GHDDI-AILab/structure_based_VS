from functools import partial
from models.bert_model import model

# seq_q = simple BERT/transformer
BertConfig = model.BertConfig
model_seq_1 = model.BertModel_embed
model_seq_1_CL = model.BertModel_embed_CL
