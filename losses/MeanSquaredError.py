from .base import LossBase
import numpy as np

class MeanSquaredError(LossBase):
    """Mean Squared Error loss."""

    def compute(self, predicted_y: np.ndarray, true_y: np.ndarray) -> float:
        """Compute the Mean Squared Error loss."""
        return np.mean(np.square(predicted_y - true_y))

    def gradient(self, predicted_y: np.ndarray, true_y: np.ndarray) -> np.ndarray:
        """Compute the gradient of MSE with respect to predictions."""
        m = predicted_y.size if predicted_y.size else 1
        return 2 * (predicted_y - true_y) / m
