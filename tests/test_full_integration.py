import unittest
import numpy as np
from blueprints.sequential import SequentialBlueprint
from construction.builder import ModelBuilder
from bricks.dense import DenseBrick
from bricks.activation import ActivationBrick
from bricks.batchnorm import BatchNormBrick
from bricks.dropout import DropoutBrick
from activations.relu import Relu
from activations.linear import Linear
from losses.MeanSquaredError import MeanSquaredError
from optimizers.adam import Adam

class TestFullIntegration(unittest.TestCase):
    def test_complex_model_training(self):
        # 1. Data Generation (Non-linear regression)
        # y = x^2 + noise
        np.random.seed(42)
        X = np.random.randn(200, 5)
        Y = np.sum(X**2, axis=1, keepdims=True) + np.random.randn(200, 1) * 0.1
        
        # Split into train/val
        X_train, X_val = X[:160], X[160:]
        Y_train, Y_val = Y[:160], Y[160:]
        
        # 2. Model Construction
        # Architecture: Input(5) -> Dense(10) -> BN -> Relu -> Dropout -> Dense(1)
        model = SequentialBlueprint()
        model.add(DenseBrick(5, 10)) # Linear Dense
        model.add(BatchNormBrick(10))        # Batch Norm
        model.add(ActivationBrick(Relu()))
        model.add(DropoutBrick(rate=0.2))
        model.add(DenseBrick(10, 1))
        model.add(ActivationBrick(Linear()))
        
        # 3. Optimizer with Stability Features
        optimizer = Adam(
            learning_rate=0.01,
            clip_norm=1.0,
            lr_decay_rate=0.001
        )
        
        loss_fn = MeanSquaredError()
        
        # 4. Trainer Setup
        builder = ModelBuilder(model, optimizer, loss_fn)
        
        # 5. Training with Early Stopping
        print("\nStarting Full Integration Test...")
        builder.fit(
            X_train, Y_train,
            x_val=X_val,
            y_val=Y_val,
            patience=10,
            early_stopping=True,
            epochs=200,
            batch_size=32,
            verbose=True
        )
        
        # 6. Verification
        # Check if loss decreased significantly
        
        y_pred = builder.predict(X_val)
        final_loss = loss_fn.compute(y_pred, Y_val)
        print(f"Final Validation Loss: {final_loss}")
        
        # Just check if it's reasonable (e.g. < 6.0) given the noise and complexity
        self.assertLess(final_loss, 6.0, "Model should learn something")
        
        # Check if BN params moved from initial values
        bn_layer = model.layers[1]
        self.assertFalse(np.allclose(bn_layer.gamma, 1.0), "BN Gamma should be updated")
        self.assertFalse(np.allclose(bn_layer.beta, 0.0), "BN Beta should be updated")

if __name__ == '__main__':
    unittest.main()
