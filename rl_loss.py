import torch
from typing import Optional

class RLLoss(torch.nn.Module):
    def __init__(self,
                 clipping_coef: Optional[float],
                 is_ppo: bool):
        super().__init__()
        self.clipping_coef = clipping_coef
        self.is_ppo = is_ppo
        

    def forward(self,
                rewards: torch.Tensor,
                log_paramterized_policy_probs: torch.Tensor,
                log_base_policy_probs: torch.Tensor,
                ):
        return -torch.mean(self._w(rewards, log_paramterized_policy_probs, log_base_policy_probs) * rewards)
        
    
    def _w(self, rewards: torch.Tensor, log_paramterized_policy_probs: torch.Tensor,
            log_base_policy_probs: torch.Tensor) -> torch.Tensor:
        weights = torch.exp(log_paramterized_policy_probs - log_base_policy_probs)
        if self.is_ppo and self.clipping_coef is not None:
            return torch.where(
                rewards > 0,
                torch.clamp(weights, max=1 + self.clipping_coef), 
                torch.clamp(weights, min=1 - self.clipping_coef)
                )
        else:
            return weights

    
    

    
    