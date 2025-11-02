from .base import ActivationBase
import numpy as np

class Sigmoid(ActivationBase):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def _derivative(self, activated_x: np.ndarray) -> np.ndarray:
        return activated_x * (1 - activated_x)