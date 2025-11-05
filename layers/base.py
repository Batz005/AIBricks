# layers/base.py
from abc import ABC, abstractmethod
import numpy as np

class LayerBase(ABC):
    def __init__(self):
        self.A_prev = None
        self.Z = None
        self.A = None

    @abstractmethod
    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """Compute the forward pass and return output."""
        pass

    @abstractmethod
    def backward(self, dA: np.ndarray, learning_rate: float) -> np.ndarray:
        """Compute the backward pass and return dA_prev."""
        pass

    def __str__(self):
        return self.__class__.__name__
