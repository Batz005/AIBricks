# models/dense_brick.py
import numpy as np
from models.base import NetworkBase
from layers import LayerBase
from losses import LossBase

class DenseBrick(NetworkBase):
    def __init__(self, layers: list[LayerBase] = None):
        super().__init__()
        self.activations: list[np.ndarray] = []
        if layers:
            for layer in layers:
                self.add(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activations = [x]  # store input as first activation
        for layer in self.layers:
            x = layer.forward(x)
            self.activations.append(x)
        return x

    def backward(self, loss_grad: np.ndarray, learning_rate: float):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad

    def compute_loss_and_grad(self, y_true: np.ndarray, y_pred: np.ndarray, loss_fn: LossBase):
        loss = loss_fn.compute(y_pred, y_true)
        grad = loss_fn.gradient(y_pred, y_true)
        return loss, grad

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: LossBase,
        epochs: int = 1000,
        learning_rate: float = 0.01,
        verbose: bool = True,
    ):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss, grad = self.compute_loss_and_grad(y, y_pred, loss_fn)
            self.backward(grad, learning_rate)
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
