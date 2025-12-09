import unittest
import numpy as np
from bricks.batchnorm import BatchNormBrick
from optimizers.sgd import SGD

class TestBatchNorm(unittest.TestCase):
    def test_forward_normalization(self):
        # Input with mean 10 and var 4
        X = np.random.randn(100, 5) * 2 + 10
        bn = BatchNormBrick(num_features=5)
        
        # Forward (Training)
        out = bn.forward(X, training=True)
        
        # Output should have mean ~0 and var ~1
        self.assertTrue(np.allclose(np.mean(out, axis=0), 0, atol=0.1))
        self.assertTrue(np.allclose(np.var(out, axis=0), 1, atol=0.1))
        
        # Running stats should be updated (momentum 0.9)
        # running_mean starts at 0. Updated: 0.9*0 + 0.1*10 = 1.0
        self.assertTrue(np.allclose(bn.running_mean, 1.0, atol=0.2))

    def test_parameter_update(self):
        bn = BatchNormBrick(num_features=1)
        bn.dGamma = np.array([[0.1]])
        bn.dBeta = np.array([[0.1]])
        
        optimizer = SGD(learning_rate=1.0)
        optimizer.step([bn])
        
        # Gamma starts at 1.0, should become 0.9
        self.assertAlmostEqual(bn.gamma[0,0], 0.9)
        # Beta starts at 0.0, should become -0.1
        self.assertAlmostEqual(bn.beta[0,0], -0.1)

    def test_inference_mode(self):
        bn = BatchNormBrick(num_features=1)
        # Manually set running stats
        bn.running_mean = np.array([[10.0]])
        bn.running_var = np.array([[4.0]]) # std = 2
        
        # Input 12.0. (12 - 10) / 2 = 1.0
        X = np.array([[12.0]])
        out = bn.forward(X, training=False)
        
        self.assertAlmostEqual(out[0,0], 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
