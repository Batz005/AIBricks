from abc import ABC, abstractmethod
from typing import List
import numpy as np
from bricks import BrickBase

class OptimizerBase(ABC):
    def __init__(self, learning_rate: float = 0.01, lr_decay_rate: float = 0.0, clip_norm: float = 1.0):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.clip_norm = clip_norm
        self.epsilon = 1e-8
        self.t = 0

    @abstractmethod
    def update(self, layer: BrickBase):
        """Update parameters of a single layer."""
        pass

    def step(self, layers: List[BrickBase]):
        """Update parameters of all layers."""
        self.t += 1
        self.learning_rate = self.initial_lr * (1 / (1 + self.lr_decay_rate * self.t))
        total_norm = 0.0

        for layer in layers:
            # Collect all gradients
            grads = []
            if hasattr(layer, 'dW') and layer.dW is not None: grads.append(layer.dW)
            if hasattr(layer, 'db') and layer.db is not None: grads.append(layer.db)
            if hasattr(layer, 'dGamma') and layer.dGamma is not None: grads.append(layer.dGamma)
            if hasattr(layer, 'dBeta') and layer.dBeta is not None: grads.append(layer.dBeta)
            
            for g in grads:
                total_norm += np.linalg.norm(g)**2
        
        total_norm = np.sqrt(total_norm)
        scale = 1.0
        if total_norm > self.clip_norm:
            scale = self.clip_norm / total_norm

        for layer in layers:
            # Apply scaling
            if scale < 1.0:
                if hasattr(layer, 'dW') and layer.dW is not None: layer.dW *= scale
                if hasattr(layer, 'db') and layer.db is not None: layer.db *= scale
                if hasattr(layer, 'dGamma') and layer.dGamma is not None: layer.dGamma *= scale
                if hasattr(layer, 'dBeta') and layer.dBeta is not None: layer.dBeta *= scale
            
            self.update(layer)
