from .base import ActivationBase
import numpy as np

class Linear(ActivationBase):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def _derivative(self, activated_x: np.ndarray) -> np.ndarray:
        return np.ones_like(activated_x)
