# layers/dense.py
import numpy as np
from activations import ActivationBase
from initializers import InitializerBase, Auto
from regularizers import RegularizerBase
from .base import LayerBase

class Dense(LayerBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: ActivationBase,
        initializer: InitializerBase | None = None,
        regularizer: RegularizerBase | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.regularizer = regularizer

        # Use autodetect if no initializer provided
        self.initializer = initializer or Auto(activation.__class__.__name__)

        # Initialize weights and biases
        self.W = self.initializer((output_dim, input_dim))  # shape: (out, in)
        self.b = np.zeros((1, output_dim))  # shape: (1, out)

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """
        A_prev: shape (batch_size, input_dim)
        Returns: shape (batch_size, output_dim)
        """
        self.A_prev = A_prev
        self.Z = A_prev @ self.W.T + self.b  # (batch, out)
        self.A = self.activation.forward(self.Z)
        return self.A

    def backward(self, dA: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        dA: gradient from next layer, shape (batch_size, output_dim)
        Returns: dA_prev to pass to previous layer
        """
        m = dA.shape[0]

        dZ = dA * self.activation.backward(self.Z, use_cached_output=False)
        dW = (dZ.T @ self.A_prev) / m  # (out, in)
        db = np.sum(dZ, axis=0, keepdims=True) / m  # (1, out)

        # Apply regularization if any
        if self.regularizer:
            dW += self.regularizer.gradient(self.W)

        dA_prev = dZ @ self.W  # (batch, in)

        # Update weights and biases
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dA_prev
