import numpy as np
from .base import BlueprintBase
from bricks import BrickBase

class SequentialBlueprint(BlueprintBase):
    def __init__(self, layers: list[BrickBase] = None):
        super().__init__()
        if layers:
            for layer in layers:
                self.add(layer)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, loss_grad: np.ndarray):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
