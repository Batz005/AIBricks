import unittest
import numpy as np
from bricks.dropout import DropoutBrick

class TestDropout(unittest.TestCase):
    def test_dropout_training(self):
        # Create dropout layer with 50% drop rate
        layer = DropoutBrick(rate=0.5)
        X = np.ones((100, 100))
        
        # Run twice
        out1 = layer.forward(X, training=True)
        out2 = layer.forward(X, training=True)
        
        # Should be different (random masks)
        self.assertFalse(np.array_equal(out1, out2), "Dropout should be random during training")
        
        # Check if approx 50% are zero
        zero_fraction = np.mean(out1 == 0)
        self.assertAlmostEqual(zero_fraction, 0.5, delta=0.1, msg="Should drop approx 50% of neurons")
        
        # Check scaling: Non-zero elements should be 1 / 0.5 = 2
        non_zeros = out1[out1 != 0]
        self.assertTrue(np.allclose(non_zeros, 2.0), "Active neurons should be scaled")

    def test_dropout_inference_determinism(self):
        dropout = DropoutBrick(rate=0.5)
        X = np.ones((10, 10))
        
        out1 = dropout.forward(X, training=False)
        out2 = dropout.forward(X, training=False)
        
        # Should be identical and equal to input
        self.assertTrue(np.array_equal(out1, out2), "Dropout should be deterministic during inference")
        self.assertTrue(np.array_equal(out1, X), "Dropout should pass input unchanged during inference")

if __name__ == '__main__':
    unittest.main()
