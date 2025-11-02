from .base import ActivationBase
import numpy as np

class Relu(ActivationBase):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _derivative(self, activated_x: np.ndarray) -> np.ndarray:
        return (activated_x > 0).astype(float)