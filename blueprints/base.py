# models/base.py
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from bricks import BrickBase
from losses import LossBase

from optimizers import OptimizerBase

class BlueprintBase(ABC):
    def __init__(self):
        self.layers: List[BrickBase] = []

    def add(self, layer: BrickBase):
        self.layers.append(layer)

    @abstractmethod
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dLoss: np.ndarray) -> np.ndarray:
        pass

    def __str__(self):
        return self.__class__.__name__
