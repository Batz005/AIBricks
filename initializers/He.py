import numpy as np
from .base import InitializerBase

class He(InitializerBase):
    """
    He Initialization (Kaiming Initialization).
    Best suited for ReLU and similar activations.
    Weights are drawn from N(0, 2 / fan_in)
    """

    def initialize(self, shape: tuple[int, int]) -> np.ndarray:
        fan_out, fan_in = shape
        return np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
