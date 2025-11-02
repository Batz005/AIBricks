import numpy as np
from .base import InitializerBase

class Xavier(InitializerBase):
    """
    Xavier/Glorot initialization for tanh/sigmoid activations.
    Draws values from a uniform distribution bounded by ±sqrt(6 / (fan_in + fan_out)).
    """
    def initialize(self, shape: tuple) -> np.ndarray:
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)