import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import copy


class AdversarialTrainer:
    
    def __init__(self, 
                 epsilon: float = 0.01,
                 learning_rate: float = 1e-6,
                 max_iterations: int = 3,
                 norm_type: str = 'l2'):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.norm_type = norm_type
        
    def generate_adversarial(self,
                            tuples: List[Dict],
                            encoder: nn.Module,
                            matcher: nn.Module,
                            training: bool = True) -> List[Tuple[Dict, Dict]]:
        adversarial_pairs = []
        
        encoder_mode = encoder.training
        matcher_mode = matcher.training
        
        if training:
            encoder.eval() 
            matcher.train()
        
        for tuple_data in tuples:
            adv_tuple = self._generate_single_adversarial(
                tuple_data, encoder, matcher
            )
            adversarial_pairs.append((tuple_data, adv_tuple))
        
        encoder.train(encoder_mode)
        matcher.train(matcher_mode)
        
        return adversarial_pairs
    
    def _generate_single_adversarial(self,
                                    tuple_data: Dict,
                                    encoder: nn.Module,
                                    matcher: nn.Module) -> Dict:
        original_emb = encoder(tuple_data)
        
        perturbed_emb = original_emb.clone().detach().requires_grad_(True)
        
        score = matcher(perturbed_emb, original_emb.detach(),
                       tuple_data['mask'], tuple_data['mask'])
        
        loss = -torch.log(score + 1e-8)

        loss.backward()
        
        grad = perturbed_emb.grad.data
        

        perturbation = self._compute_perturbation(grad)
        
        adversarial_emb = original_emb + perturbation
        
        ##
        adv_tuple = copy.deepcopy(tuple_data)
        adv_tuple['_adversarial_embedding'] = adversarial_emb.detach()
        
        return adv_tuple
    
    def _compute_perturbation(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.norm_type == 'l2':
            grad_norm = torch.norm(gradient, p=2)
            if grad_norm > 1e-8:
                perturbation = -self.epsilon * gradient / grad_norm
            else:
                perturbation = torch.zeros_like(gradient)
                
        elif self.norm_type == 'linf':
            perturbation = -self.epsilon * torch.sign(gradient)
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")
        
        return perturbation
    
    def generate_adversarial_iterative(self,
                                      tuples: List[Dict],
                                      encoder: nn.Module,
                                      matcher: nn.Module,
                                      num_iterations: Optional[int] = None) -> List[Tuple[Dict, Dict]]:
        if num_iterations is None:
            num_iterations = self.max_iterations
        
        adversarial_pairs = []
        step_size = self.epsilon / num_iterations
        
        encoder.eval()
        matcher.train()
        
        for tuple_data in tuples:
            ##
            original_emb = encoder(tuple_data)
            perturbed_emb = original_emb.clone().detach()
            
            for _ in range(num_iterations):
                perturbed_emb.requires_grad_(True)
                
                score = matcher(perturbed_emb, original_emb,
                              tuple_data['mask'], tuple_data['mask'])
                loss = -torch.log(score + 1e-8)
                
                loss.backward()
                grad = perturbed_emb.grad.data
                
                if self.norm_type == 'l2':
                    grad_norm = torch.norm(grad, p=2)
                    if grad_norm > 1e-8:
                        step = -step_size * grad / grad_norm
                    else:
                        step = torch.zeros_like(grad)
                else:  
                    step = -step_size * torch.sign(grad)
                
                perturbed_emb = perturbed_emb.detach() + step
                
                perturbation = perturbed_emb - original_emb
                perturbation = self._project_to_epsilon_ball(perturbation)
                perturbed_emb = original_emb + perturbation
            
            adv_tuple = copy.deepcopy(tuple_data)
            adv_tuple['_adversarial_embedding'] = perturbed_emb.detach()
            adversarial_pairs.append((tuple_data, adv_tuple))
        
        return adversarial_pairs
    
    def _project_to_epsilon_ball(self, perturbation: torch.Tensor) -> torch.Tensor:
        if self.norm_type == 'l2':
            norm = torch.norm(perturbation, p=2)
            if norm > self.epsilon:
                perturbation = perturbation * (self.epsilon / norm)
        elif self.norm_type == 'linf':
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        
        return perturbation
    
    def compute_adversarial_loss(self,
                                adversarial_pairs: List[Tuple],
                                encoder: nn.Module,
                                matcher: nn.Module) -> torch.Tensor:

        total_loss = 0.0
        
        for original_tuple, adv_tuple in adversarial_pairs:
            if '_adversarial_embedding' in adv_tuple:
                adv_emb = adv_tuple['_adversarial_embedding']
                orig_emb = encoder(original_tuple)
            else:
                orig_emb = encoder(original_tuple)
                adv_emb = encoder(adv_tuple)
            
            score = matcher(orig_emb, adv_emb,
                          original_tuple['mask'], adv_tuple['mask'])
            
            loss = -torch.log(score + 1e-8)
            total_loss += loss
        
        avg_loss = total_loss / len(adversarial_pairs)
        return avg_loss
    
    def get_perturbation_stats(self,
                              adversarial_pairs: List[Tuple],
                              encoder: nn.Module) -> Dict[str, float]:
        perturbation_norms = []
        
        for original_tuple, adv_tuple in adversarial_pairs:
            if '_adversarial_embedding' in adv_tuple:
                adv_emb = adv_tuple['_adversarial_embedding']
                orig_emb = encoder(original_tuple)
            else:
                orig_emb = encoder(original_tuple)
                adv_emb = encoder(adv_tuple)
            
            perturbation = adv_emb - orig_emb
            norm = torch.norm(perturbation, p=2).item()
            perturbation_norms.append(norm)
        
        return {
            'mean_perturbation_norm': sum(perturbation_norms) / len(perturbation_norms),
            'max_perturbation_norm': max(perturbation_norms),
            'min_perturbation_norm': min(perturbation_norms),
            'epsilon': self.epsilon
        }


class AdversarialTrainingScheduler:
    
    def __init__(self,
                 initial_epsilon: float = 0.001,
                 final_epsilon: float = 0.01,
                 warmup_epochs: int = 5,
                 total_epochs: int = 30):

        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def step(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            epsilon = (self.initial_epsilon + 
                      (self.final_epsilon - self.initial_epsilon) * 
                      (self.current_epoch / self.warmup_epochs))
        else:
            epsilon = self.final_epsilon
        
        self.current_epoch += 1
        return epsilon
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_epoch = 0