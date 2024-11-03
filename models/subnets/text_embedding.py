import torch
import torch.nn as nn

import sys
sys.path.append('..')
from utils.tools import *

device=get_device()

# Get text embeddings
class GetTextEmbedding(nn.Module):
    def __init__(self, tokenizer, text_model):
        super(GetTextEmbedding, self).__init__()
        self.tokenizer = tokenizer
        self.text_model = text_model

    def forward(self, text_inputs):
        text_embeddings = self.tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True,
                                         max_length=80).to(device)
        text_output = self.text_model(**text_embeddings)
        text_seq_embedding, text_cls_embeddings = text_output[0], text_output[1]
        return text_seq_embedding, text_cls_embeddings
