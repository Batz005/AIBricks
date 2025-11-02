"""
L2 regularization module.

This module provides an L2 regularizer class that applies ridge-style regularization
to model weights, helping to prevent large weights and encouraging smoothness.
"""
from .base import RegularizerBase
import numpy as np

class L2(RegularizerBase):
    """
    L2 regularizer (ridge-style).
    penalty = 0.5 * lambda_ * sum(w^2)
    gradient = lambda_ * w
    This helps prevent large weights and encourages smoothness.
    """

    def __init__(self, lambda_: float = 0.01):
        self.lambda_ = lambda_

    def penalty(self, weights):
        # Return the scalar L2 penalty: 0.5 * lambda * sum(weights^2)
        return 0.5 * self.lambda_ * np.sum(weights**2)

    def gradient(self, weights):
        # Return the gradient of the L2 penalty: lambda * weights
        return self.lambda_ * weights