from .base import OptimizerBase
from .base import OptimizerBase
from bricks import BrickBase
import numpy as np
from dataclasses import dataclass

@dataclass
class AdamState:
    m_w: int = 0
    v_w: int = 0
    m_b: int = 0
    v_b: int = 0
    m_gamma: int = 0
    v_gamma: int = 0
    m_beta: int = 0
    v_beta: int = 0


class Adam(OptimizerBase):
    def __init__(self, learning_rate: float = 0.01,clip_norm: float = 1.0, lr_decay_rate: float = 0.999, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__(learning_rate, lr_decay_rate, clip_norm)
        self.layer_states = {}
        self.beta1 = beta1
        self.beta2 = beta2
        

    def step(self, layers):
        super().step(layers)

    def update(self, layer: BrickBase):
        if layer not in self.layer_states:
            self.layer_states[layer] = AdamState()
        
        
        if hasattr(layer, 'W') and hasattr(layer, 'dW') and layer.dW is not None:

            self.layer_states[layer].m_w = self.beta1 * self.layer_states[layer].m_w + (1 - self.beta1) * layer.dW
            self.layer_states[layer].v_w = self.beta2 * self.layer_states[layer].v_w + (1 - self.beta2) * (layer.dW ** 2)

            m_hat_w = self.layer_states[layer].m_w / (1 - self.beta1 ** self.t)
            v_hat_w = self.layer_states[layer].v_w / (1 - self.beta2 ** self.t)

            layer.W -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        
        if hasattr(layer, 'b') and hasattr(layer, 'db') and layer.db is not None:
            self.layer_states[layer].m_b = self.beta1 * self.layer_states[layer].m_b + (1 - self.beta1) * layer.db
            self.layer_states[layer].v_b = self.beta2 * self.layer_states[layer].v_b + (1 - self.beta2) * (layer.db ** 2)

            m_hat_b = self.layer_states[layer].m_b / (1 - self.beta1 ** self.t)
            v_hat_b = self.layer_states[layer].v_b / (1 - self.beta2 ** self.t)

            layer.b -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        if hasattr(layer, 'gamma') and hasattr(layer, 'dGamma') and layer.dGamma is not None:
            self.layer_states[layer].m_gamma = self.beta1 * self.layer_states[layer].m_gamma + (1 - self.beta1) * layer.dGamma
            self.layer_states[layer].v_gamma = self.beta2 * self.layer_states[layer].v_gamma + (1 - self.beta2) * (layer.dGamma ** 2)

            m_hat_gamma = self.layer_states[layer].m_gamma / (1 - self.beta1 ** self.t)
            v_hat_gamma = self.layer_states[layer].v_gamma / (1 - self.beta2 ** self.t)

            layer.gamma -= self.learning_rate * m_hat_gamma / (np.sqrt(v_hat_gamma) + self.epsilon)

        if hasattr(layer, 'beta') and hasattr(layer, 'dBeta') and layer.dBeta is not None:
            self.layer_states[layer].m_beta = self.beta1 * self.layer_states[layer].m_beta + (1 - self.beta1) * layer.dBeta
            self.layer_states[layer].v_beta = self.beta2 * self.layer_states[layer].v_beta + (1 - self.beta2) * (layer.dBeta ** 2)

            m_hat_beta = self.layer_states[layer].m_beta / (1 - self.beta1 ** self.t)
            v_hat_beta = self.layer_states[layer].v_beta / (1 - self.beta2 ** self.t)

            layer.beta -= self.learning_rate * m_hat_beta / (np.sqrt(v_hat_beta) + self.epsilon)