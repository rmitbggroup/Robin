import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AIJNet(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: List[int]):
        super(AIJNet, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        mlp_layers = []
        input_dim = embedding_dim * 2 
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor,
                mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        concat_emb = torch.cat([emb1, emb2], dim=-1)
        
        attended_emb = self._self_attention(concat_emb, 
                                           torch.cat([mask1, mask2], dim=-1))
        
        logits = self.mlp(attended_emb)
        score = torch.sigmoid(logits)
        
        return score
    
    def _self_attention(self, embeddings: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
        Q = self.query_proj(embeddings)
        K = self.key_proj(embeddings)
        V = self.value_proj(embeddings)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        
        mask = mask.unsqueeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        return attended