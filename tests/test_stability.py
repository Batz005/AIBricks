import unittest
import numpy as np
from optimizers.sgd import SGD
from bricks import BrickBase
from blueprints.sequential import SequentialBlueprint
from construction.builder import ModelBuilder
from bricks.dense import DenseBrick
from bricks.activation import ActivationBrick
from activations.linear import Linear
from losses.MeanSquaredError import MeanSquaredError

class TestStability(unittest.TestCase):
    def test_gradient_clipping(self):
        # Create a layer with massive gradients
        class ExplodingLayer(BrickBase):
            def __init__(self):
                self.dW = np.array([[1000.0]])
                self.db = np.array([[1000.0]])
            def forward(self, x, training=True): return x
            def backward(self, da): return da

        layer = ExplodingLayer()
        optimizer = SGD(learning_rate=0.1, clip_norm=1.0)
        
        # Step
        optimizer.step([layer])
        
        # Norm is sqrt(1000^2 + 1000^2) approx 1414
        # Scale should be 1 / 1414
        # New dW should be approx 1000/1414 = 0.707
        
        self.assertLess(abs(layer.dW[0,0]), 1.0)
        self.assertLess(abs(layer.db[0,0]), 1.0)

    def test_lr_decay(self):
        optimizer = SGD(learning_rate=0.1, lr_decay_rate=0.1)
        
        # Simulate 10 steps
        for _ in range(10):
            optimizer.step([])
            
        # Initial LR = 0.1
        # After 10 steps: 0.1 / (1 + 0.1 * 10) = 0.1 / 2 = 0.05
        self.assertAlmostEqual(optimizer.learning_rate, 0.05)

    def test_early_stopping(self):
        # Setup data where validation loss will likely increase or plateau
        X_train = np.random.randn(100, 1)
        Y_train = X_train * 2
        
        X_val = np.random.randn(20, 1)
        Y_val = np.random.randn(20, 1) * 10 # Random noise, uncorrelated with X_val
        
        model = SequentialBlueprint()
        model.add(DenseBrick(1, 1))
        model.add(ActivationBrick(Linear()))
        
        optimizer = SGD(learning_rate=0.01)
        loss_fn = MeanSquaredError()
        builder = ModelBuilder(model, optimizer, loss_fn)
        
        # Capture stdout to verify "Early stopping" message
        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output
        
        builder.fit(
            X_train, Y_train,
            x_val=X_val,
            y_val=Y_val,
            patience=2,
            early_stopping=True,
            epochs=50,
            verbose=True
        )
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        print("Starting Early Stopping Test:")
        print(output)
        
        self.assertIn("Early stopping", output)

if __name__ == '__main__':
    unittest.main()
