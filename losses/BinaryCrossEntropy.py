from .base import LossBase
import numpy as np

class BinaryCrossEntropy(LossBase):
    def _compute_loss(self, predicted_y: np.ndarray, true_y: np.ndarray) -> float:
        return -np.average(np.sum(true_y * np.log(predicted_y + self.epsilon) + (1 - true_y) * np.log(1 - predicted_y + self.epsilon)))


    def _compute_gradient(self, predicted_y: np.ndarray, true_y: np.ndarray) -> np.ndarray:
        pass