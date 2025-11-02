from abc import ABC, abstractmethod
import numpy as np

class LossBase(ABC):
    """Abstract base class for all loss functions."""

    @abstractmethod
    def compute(self, predicted_y: np.ndarray, true_y: np.ndarray) -> float:
        """compute the actual loss."""
        pass

    @abstractmethod
    def gradient(self, predicted_y: np.ndarray, true_y: np.ndarray) -> np.ndarray:
        """Clip values and then compute the gradient."""
        pass

    def _clip(self, y: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Safely clip predictions to prevent log(0) or division by zero."""
        return np.clip(y, epsilon, 1 - epsilon)
    
    def __str__(self):
        return self.__class__.__name__