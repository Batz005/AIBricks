from .base import RegularizerBase
import numpy as np

class ElasticNet(RegularizerBase):
    """
    ElasticNet regularizer combining L1 and L2.
    Parameterization:
      lambda_: overall regularization strength
      l1_ratio: fraction of penalty allocated to L1 (in [0, 1]).
    Common formulation:
      penalty = lambda_ * ( l1_ratio * sum|w| + (1 - l1_ratio) * 0.5 * sum(w^2) )
    Gradient:
      grad = lambda_ * ( l1_ratio * sign(w) + (1 - l1_ratio) * w )
    """
    def __init__(self, lambda_: float = 0.01, l1_ratio: float = 0.5):
        assert 0.0 <= l1_ratio <= 1.0, "l1_ratio must be in [0, 1]"
        self.lambda_ = lambda_
        self.l1_ratio = float(l1_ratio)

    def penalty(self, weights: np.ndarray) -> float:
        l1 = np.sum(np.abs(weights))
        l2 = 0.5 * np.sum(weights ** 2)
        return self.lambda_ * (self.l1_ratio * l1 + (1.0 - self.l1_ratio) * l2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        l1_grad = np.sign(weights).astype(float)
        l1_grad[weights == 0] = 0.0
        l2_grad = weights
        return self.lambda_ * (self.l1_ratio * l1_grad + (1.0 - self.l1_ratio) * l2_grad)
