import numpy as np
from .base import InitializerBase
from .He import He
from .Xavier import Xavier

class Auto(InitializerBase):
    """
    Automatically chooses an initialization strategy based on the activation function.

    Supported:
    - ReLU → He Initialization
    - Tanh, Sigmoid → Xavier Initialization
    """

    def __init__(self, activation_name: str):
        self.activation_name = activation_name.lower()
        self.strategy = self._choose_strategy()

    def _choose_strategy(self):
        if self.activation_name == 'relu':
            return He()
        elif self.activation_name in ('tanh', 'sigmoid'):
            return Xavier()
        else:
            raise ValueError(f"Unsupported activation for auto initializer: {self.activation_name}")

    def initialize(self, shape: tuple[int, int]) -> np.ndarray:
        return self.strategy.initialize(shape)