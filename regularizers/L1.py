from .base import RegularizerBase
import numpy as np

class L1(RegularizerBase):
    """
    L1 regularizer (lasso-style).
    penalty = lambda_ * sum(|w|)
    gradient (subgradient) = lambda_ * sign(w)
      - at w == 0 we return 0 (a valid subgradient choice)
    """
    def __init__(self, lambda_: float = 0.01):
        self.lambda_ = lambda_

    def penalty(self, weights: np.ndarray) -> float:
        return self.lambda_ * np.sum(np.abs(weights))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        grad = np.sign(weights).astype(float)
        # choose subgradient 0 at exact zeros (common choice)
        grad[weights == 0] = 0.0
        return self.lambda_ * grad