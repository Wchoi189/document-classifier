import torch.optim.lr_scheduler as lr_scheduler
import math


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 스캐줄러 래퍼"""
    
    def __init__(self, warmup_steps: int = 1000, max_steps: int = 10000, 
                 min_lr: float = 0.0, warmup_start_lr: float = 0.0):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
    
    def __call__(self, optimizer):
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Warmup phase
                return self.warmup_start_lr + (1.0 - self.warmup_start_lr) * step / self.warmup_steps
            else:
                # Cosine annealing phase
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return self.min_lr + (1.0 - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return lr_scheduler.LambdaLR(optimizer, lr_lambda) 