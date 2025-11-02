from abc import ABC, abstractmethod
import numpy as np

class ActivationBase(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the activation function."""
        pass
    
    @abstractmethod
    def _derivative(self, activated_x: np.ndarray) -> np.ndarray:
        """Compute the derivative from the activated output."""
        pass
    
    def backward(self, x: np.ndarray, use_cached_output: bool = False) -> np.ndarray:
        if use_cached_output:
            return self._derivative(x)
        else:
            return self._derivative(self.forward(x))