from abc import ABC, abstractmethod
import numpy as np

class InitializerBase(ABC):
    @abstractmethod
    def initialize(self, shape: tuple) -> np.ndarray:
        pass

    def __call__(self, shape: tuple) -> np.ndarray:
        return self.initialize(shape)

    def __str__(self):
        return self.__class__.__name__