# layers/dense.py
import numpy as np
from initializers import InitializerBase, Xavier
from regularizers import RegularizerBase
from .base import BrickBase

class DenseBrick(BrickBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initializer: InitializerBase | None = None,
        regularizer: RegularizerBase | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.regularizer = regularizer

        # Default to Xavier initialization
        self.initializer = initializer or Xavier()

        # Initialize weights and biases
        self.W = self.initializer((output_dim, input_dim))  # shape: (out, in)
        self.b = np.zeros((1, output_dim))  # shape: (1, out)

        # Gradients (Initialize to None)
        self.dW = None
        self.db = None

    def forward(self, A_prev: np.ndarray, training: bool = True) -> np.ndarray:
        """
        A_prev: shape (batch_size, input_dim)
        Returns: shape (batch_size, output_dim)
        """
        self.A_prev = A_prev
        # Linear pass only
        self.Z = A_prev @ self.W.T + self.b  # (batch, out)
        self.logger.debug(f"Forward: Input {A_prev.shape} -> Output {self.Z.shape}")
        return self.Z

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        dA: gradient from next layer (dL/dZ), shape (batch_size, output_dim)
        Returns: dA_prev to pass to previous layer
        """
        m = dA.shape[0]

        # dA is dZ because there is no activation here
        dZ = dA

        # Calculate Gradients
        self.dW = (dZ.T @ self.A_prev) / m  # (out, in)
        self.db = np.sum(dZ, axis=0, keepdims=True) / m  # (1, out)

        # Apply regularization if any
        if self.regularizer:
            self.dW += self.regularizer.gradient(self.W)

        dA_prev = dZ @ self.W  # (batch, in)

        return dA_prev
