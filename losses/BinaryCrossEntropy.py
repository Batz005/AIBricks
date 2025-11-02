from .base import LossBase
import numpy as np

class BinaryCrossEntropy(LossBase):
    def _compute_loss(self, predicted_y: np.ndarray, true_y: np.ndarray) -> float:
        predicted_y = self._clip(predicted_y)
        return -np.mean(
            true_y * np.log(predicted_y) +
            (1 - true_y) * np.log(1 - predicted_y)
        )


    def _compute_gradient(self, predicted_y: np.ndarray, true_y: np.ndarray) -> np.ndarray:
        predicted_y = self._clip(predicted_y)
        return (predicted_y - true_y) / (predicted_y * (1 - predicted_y))