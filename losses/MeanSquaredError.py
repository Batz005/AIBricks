from .base import LossBase
import numpy as np

class MeanSquaredError(LossBase):
    """Mean Squared Error loss."""

    def compute(self, predicted_y: np.ndarray, true_y: np.ndarray) -> float:
        """Compute the Mean Squared Error loss."""
        return np.mean(np.square(predicted_y - true_y))

    def gradient(self, predicted_y: np.ndarray, true_y: np.ndarray) -> np.ndarray:
        """Compute the gradient of MSE with respect to predictions."""
        m = predicted_y.shape[1] if predicted_y.ndim > 1 else predicted_y.size
        return 2 * (predicted_y - true_y) / m