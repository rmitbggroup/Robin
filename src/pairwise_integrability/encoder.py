import torch
import torch.nn as nn
from typing import Dict
from transformers import AutoModel, AutoTokenizer


class TupleEncoder(nn.Module):
    
    def __init__(self, pretrained_model: str = 'microsoft/deberta-v3-base',
                 embedding_dim: int = 768):
        super(TupleEncoder, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.plm = AutoModel.from_pretrained(pretrained_model)
        self.embedding_dim = embedding_dim
        
        # Freeze PLM parameters
        for param in self.plm.parameters():
            param.requires_grad = False
    
    def forward(self, tuple_data: Dict) -> torch.Tensor:
        embeddings = []
        
        for attr_value in tuple_data['attributes']:
            if attr_value is None:
                attr_emb = self._encode_null()
            else:
                attr_emb = self._encode_attribute(attr_value)
            embeddings.append(attr_emb)
        
        tuple_embedding = torch.cat(embeddings, dim=-1)
        
        return tuple_embedding
    
    def _encode_attribute(self, attr_value: str) -> torch.Tensor:
        inputs = self.tokenizer(attr_value, return_tensors='pt', 
                               truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.plm(**inputs)
        
        # Use [CLS] token embedding
        attr_embedding = outputs.last_hidden_state[:, 0, :]
        
        return attr_embedding
    
    def _encode_null(self) -> torch.Tensor:

        return self._encode_attribute("[NULL]")