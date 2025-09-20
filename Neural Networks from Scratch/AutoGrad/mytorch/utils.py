import cupy as cp
import math

def clip_grad_norm_(params, max_norm):
    """
    The norm is computed over the norms of the individual gradients of all parameters, 
    as if the norms of the individual gradients were concatenated into a 
    single vector. Gradients are modified in-place.
    """

    # Compute total norm
    total_norm = cp.sqrt(sum(cp.sum(p.grad ** 2) for p in params if p.requires_grad))
    
    # Compute scaling factor
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1.0:
        for p in params:
            if p.requires_grad:
                p.grad *= clip_coef

class CosineLRScheduler:
    def __init__(self, optimizer, max_lr, min_lr=0.0, total_steps=1000, warmup_steps=0):
        """
        Cosine learning rate scheduler.
        
        optimizer: your Adam/AdamW optimizer
        max_lr: initial / max learning rate
        min_lr: minimum learning rate at the end
        total_steps: total number of training steps
        warmup_steps: number of steps to linearly increase LR at start
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        # Update optimizer LR
        self.optimizer.lr = lr

    def get_last_lr(self):
        return self.optimizer.lr