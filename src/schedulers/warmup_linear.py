import torch.optim.lr_scheduler as lr_scheduler


class WarmupLinearScheduler:
    """Warmup + Linear Decay 스캐줄러 래퍼"""
    
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
                # Linear decay phase
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return self.min_lr + (1.0 - self.min_lr) * (1.0 - progress)
        
        return lr_scheduler.LambdaLR(optimizer, lr_lambda) 