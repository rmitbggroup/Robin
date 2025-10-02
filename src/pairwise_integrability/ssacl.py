import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .encoder import TupleEncoder
from .matcher import AIJNet
from .data_generator import DataGenerator
from .adversarial_trainer import AdversarialTrainer


class SSACL(nn.Module):
    """
    Main SSACL model for pairwise integrability judgment.
    """
    
    def __init__(self, config: Dict):
        super(SSACL, self).__init__()
        self.config = config
        
        # Initialize components
        self.encoder = TupleEncoder(
            pretrained_model=config['pretrained_model'],
            embedding_dim=config['embedding_dim']
        )
        
        self.matcher = AIJNet(
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims']
        )
        
        self.data_generator = DataGenerator(
            n_pos=config['n_pos'],
            n_neg=config['n_neg']
        )
        
        self.adversarial_trainer = AdversarialTrainer(
            epsilon=config['epsilon'],
            learning_rate=config['learning_rate']
        )
        
    def forward(self, tuple1: Dict, tuple2: Dict) -> torch.Tensor:
        """
        Forward pass for tuple pair integrability judgment.
        
        Args:
            tuple1: First tuple representation
            tuple2: Second tuple representation
            
        Returns:
            Integrability score (0 or 1)
        """
        # Encode tuples
        emb1 = self.encoder(tuple1)
        emb2 = self.encoder(tuple2)
        
        # Match embeddings
        integrability = self.matcher(emb1, emb2, 
                                     tuple1['mask'], 
                                     tuple2['mask'])
        
        return integrability
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step with contrastive learning and adversarial training.
        """
        # Generate positive and negative samples
        pos_pairs, neg_pairs = self.data_generator.generate(batch)
        
        # Compute contrastive loss
        loss = self._compute_nce_loss(pos_pairs, neg_pairs)
        
        # Generate adversarial examples
        adv_pairs = self.adversarial_trainer.generate_adversarial(
            batch, self.encoder, self.matcher
        )
        
        # Add adversarial loss
        adv_loss = self._compute_adversarial_loss(adv_pairs)
        total_loss = loss + self.config['adv_weight'] * adv_loss
        
        return {
            'total_loss': total_loss.item(),
            'contrastive_loss': loss.item(),
            'adversarial_loss': adv_loss.item()
        }
    
    def _compute_nce_loss(self, pos_pairs: List, neg_pairs: List) -> torch.Tensor:
        """
        Compute binary NCE loss.
        """
        loss = 0.0
        
        for pos_pair in pos_pairs:
            score = self.forward(pos_pair[0], pos_pair[1])
            loss += -torch.log(score + 1e-8)
        
        for neg_pair in neg_pairs:
            score = self.forward(neg_pair[0], neg_pair[1])
            loss += -torch.log(1 - score + 1e-8)
        
        return loss / (len(pos_pairs) + len(neg_pairs))
    
    def _compute_adversarial_loss(self, adv_pairs: List) -> torch.Tensor:
        """
        Compute adversarial loss.
        """
        loss = 0.0
        for adv_pair in adv_pairs:
            score = self.forward(adv_pair[0], adv_pair[1])
            loss += -torch.log(score + 1e-8)
        
        return loss / len(adv_pairs)