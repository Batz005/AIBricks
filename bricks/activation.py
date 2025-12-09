import numpy as np
from .base import BrickBase
from activations import ActivationBase

class ActivationBrick(BrickBase):
    def __init__(self, activation_fn: ActivationBase):
        super().__init__()
        self.activation_fn = activation_fn
        self.Z = None

    def forward(self, A_prev: np.ndarray, training: bool = True) -> np.ndarray:
        self.Z = A_prev
        return self.activation_fn.forward(A_prev)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA * self.activation_fn.backward(self.Z)
