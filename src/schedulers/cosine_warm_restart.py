import torch.optim.lr_scheduler as lr_scheduler


class CosineWarmRestartScheduler:
    """Cosine Annealing with Warm Restarts 스캐줄러 래퍼"""
    
    def __init__(self, T_0: int = 10, T_mult: int = 2, eta_min: float = 0.0):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
    
    def __call__(self, optimizer):
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min
        ) 