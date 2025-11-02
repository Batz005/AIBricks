from .base import ActivationBase
import numpy as np

class Tanh(ActivationBase):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def _derivative(self, activated_x: np.ndarray) -> np.ndarray:
        return 1 - (activated_x**2)