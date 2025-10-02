"""
Data generator with augmentation and negative sampling.
"""

import random
import copy
from typing import List, Dict, Tuple
from ..utils.data_augmentation import (
    attribute_removal, attribute_substitution,
    word_removal, word_substitution, word_swapping,
    character_typo
)


class DataGenerator:  
    def __init__(self, n_pos: int = 6, n_neg: int = 20):
        self.n_pos = n_pos
        self.n_neg = n_neg
        
        # Perturbation functions
        self.perturbation_funcs = [
            attribute_removal,
            attribute_substitution,
            word_removal,
            word_substitution,
            word_swapping,
            character_typo
        ]
    
    def generate(self, batch: Dict) -> Tuple[List, List]:
        """
        Generate positive and negative instances.
        
        Returns:
            Tuple of (positive_pairs, negative_pairs)
        """
        positive_pairs = []
        negative_pairs = []
        
        tuples = batch['tuples']
        
        for tuple_data in tuples:
            for _ in range(self.n_pos):
                perturbed = self._apply_perturbation(tuple_data)
                positive_pairs.append((tuple_data, perturbed))
            
            for _ in range(self.n_neg):
                neg_tuple = random.choice(tuples)
                if neg_tuple != tuple_data:
                    negative_pairs.append((tuple_data, neg_tuple))
        
        return positive_pairs, negative_pairs
    
    def _apply_perturbation(self, tuple_data: Dict) -> Dict:
        perturbed = copy.deepcopy(tuple_data)
        
        func = random.choice(self.perturbation_funcs)
        perturbed = func(perturbed)
        
        return perturbed