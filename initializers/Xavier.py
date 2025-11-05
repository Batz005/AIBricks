import numpy as np
from .base import InitializerBase

class Xavier(InitializerBase):
    """
    Xavier/Glorot initialization for tanh/sigmoid activations.
    Draws values from a uniform distribution bounded by ±sqrt(6 / (fan_in + fan_out)).
    """
    def initialize(self, shape: tuple[int, int]) -> np.ndarray:
        fan_out, fan_in = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_out, fan_in))
