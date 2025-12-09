import numpy as np
from .base import BrickBase

class BatchNormBrick(BrickBase):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        # Gradients
        self.dGamma = None
        self.dBeta = None
        
        # Running stats (for inference)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Cache for backward pass
        self.cache = None

    def forward(self, Z, training=True):
        # Z shape: (batch_size, num_features)
        
        if training:
            # Calculate batch stats
            mean = np.mean(Z, axis=0, keepdims=True)
            var = np.var(Z, axis=0, keepdims=True)
            
            # Update running stats (exponential moving average)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Normalize
            Z_centered = Z - mean
            std_inv = 1.0 / np.sqrt(var + self.epsilon)
            Z_norm = Z_centered * std_inv
            
            # Cache for backward
            self.cache = (Z_centered, std_inv, Z_norm)
        else:
            # Use running stats
            Z_norm = (Z - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            
        # Scale and Shift
        out = self.gamma * Z_norm + self.beta
        return out

    def backward(self, dOut):
        # Retrieve cache
        Z_centered, std_inv, Z_norm = self.cache
        N = dOut.shape[0]
        
        # Gradients for Gamma and Beta
        self.dGamma = np.sum(dOut * Z_norm, axis=0, keepdims=True)
        self.dBeta = np.sum(dOut, axis=0, keepdims=True)
        
        # Gradient for Input (Z) - The scary part
        dZ_norm = dOut * self.gamma
        dVar = np.sum(dZ_norm * Z_centered * -0.5 * std_inv**3, axis=0, keepdims=True)
        dMean = np.sum(dZ_norm * -std_inv, axis=0, keepdims=True) + dVar * np.mean(-2.0 * Z_centered, axis=0, keepdims=True)
        
        dZ = (dZ_norm * std_inv) + (dVar * 2 * Z_centered / N) + (dMean / N)
        return dZ