from .base import OptimizerBase
from .base import OptimizerBase
from bricks import BrickBase

class SGD(OptimizerBase):
    def __init__(self, learning_rate: float = 0.01, **kwargs):
        super().__init__(learning_rate, **kwargs)

    def update(self, layer: BrickBase):
        if hasattr(layer, 'W') and hasattr(layer, 'dW') and layer.dW is not None:
            layer.W -= self.learning_rate * layer.dW
        
        if hasattr(layer, 'b') and hasattr(layer, 'db') and layer.db is not None:
            layer.b -= self.learning_rate * layer.db

        if hasattr(layer, 'gamma') and hasattr(layer, 'dGamma') and layer.dGamma is not None:
            layer.gamma -= self.learning_rate * layer.dGamma

        if hasattr(layer, 'beta') and hasattr(layer, 'dBeta') and layer.dBeta is not None:
            layer.beta -= self.learning_rate * layer.dBeta
