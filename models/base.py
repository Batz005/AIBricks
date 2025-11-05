# models/base.py
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from layers import LayerBase
from losses import LossBase

class NetworkBase(ABC):
    def __init__(self):
        self.layers: List[LayerBase] = []

    def add(self, layer: LayerBase):
        self.layers.append(layer)

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dLoss: np.ndarray, learning_rate: float) -> np.ndarray:
        pass

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        loss_fn: LossBase,
        epochs: int,
        learning_rate: float,
        verbose: bool = True
    ):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __str__(self):
        return self.__class__.__name__
