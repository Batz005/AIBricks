import numpy as np
from .base import BrickBase

class DropoutBrick(BrickBase):
    def __init__(self, rate: float = 0.5):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, A_prev: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=A_prev.shape) / (1 - self.rate)
            return A_prev * self.mask
        else:
            return A_prev
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA * self.mask