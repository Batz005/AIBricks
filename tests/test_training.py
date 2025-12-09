import unittest
import numpy as np
from blueprints.sequential import SequentialBlueprint
from construction.builder import ModelBuilder
from bricks.dense import DenseBrick
from bricks.activation import ActivationBrick
from activations.linear import Linear
from losses.MeanSquaredError import MeanSquaredError
from optimizers.sgd import SGD

class TestTraining(unittest.TestCase):
    def test_sgd_training_reduces_loss(self):
        # Simple regression problem: y = 2x
        X = np.array([[1], [2], [3], [4]])
        Y = np.array([[2], [4], [6], [8]])

        model = SequentialBlueprint()
        # Single neuron with Linear activation
        model.add(DenseBrick(1, 1))
        model.add(ActivationBrick(Linear()))

        loss_fn = MeanSquaredError()
        optimizer = SGD(learning_rate=0.01)
        builder = ModelBuilder(model, optimizer, loss_fn)

        # Initial loss
        y_pred = builder.predict(X)
        initial_loss = loss_fn.compute(y_pred, Y)

        # Train
        builder.fit(X, Y, epochs=100, batch_size=4, verbose=False)

        # Final loss
        y_pred = builder.predict(X)
        final_loss = loss_fn.compute(y_pred, Y)

        print(f"SGD Initial Loss: {initial_loss}, Final Loss: {final_loss}")
        self.assertLess(final_loss, initial_loss, "Loss should decrease after training")

    def test_adam_training_reduces_loss(self):
        from optimizers.adam import Adam
        # Simple regression problem: y = 2x
        X = np.array([[1], [2], [3], [4]])
        Y = np.array([[2], [4], [6], [8]])

        model = SequentialBlueprint()
        # Single neuron with Linear activation
        model.add(DenseBrick(1, 1))
        model.add(ActivationBrick(Linear()))

        loss_fn = MeanSquaredError()
        optimizer = Adam(learning_rate=0.1)
        builder = ModelBuilder(model, optimizer, loss_fn)

        # Initial loss
        y_pred = builder.predict(X)
        initial_loss = loss_fn.compute(y_pred, Y)

        # Train
        builder.fit(X, Y, epochs=100, batch_size=4, verbose=False)

        # Final loss
        y_pred = builder.predict(X)
        final_loss = loss_fn.compute(y_pred, Y)

        print(f"Adam Initial Loss: {initial_loss}, Final Loss: {final_loss}")
        self.assertLess(final_loss, initial_loss, "Loss should decrease after training")

if __name__ == '__main__':
    unittest.main()
