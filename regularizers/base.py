from abc import ABC, abstractmethod
import numpy as np

class RegularizerBase(ABC):
    """Abstract base class for regularizers."""

    @abstractmethod
    def penalty(self, weights: np.ndarray) -> float:
        """
        Return the scalar penalty for the given weights (sum over all elements).
        This value is added to the loss (often scaled already by lambda).
        """
        pass

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Return the gradient (same shape as weights) of the penalty term w.r.t. weights.
        This is added to the usual dL/dW during backprop.
        """
        pass

    def __str__(self):
        return self.__class__.__name__