import numpy as np
from blueprints.base import BlueprintBase
from losses import LossBase
from optimizers import OptimizerBase
from utils.logger import get_logger

class ModelBuilder:
    def __init__(self, model: BlueprintBase, optimizer: OptimizerBase, loss_fn: LossBase):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = {'loss': [], 'val_loss': []}
        self.logger = get_logger("ModelBuilder")

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        x_val: np.ndarray = None,
        y_val: np.ndarray = None,
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 0.001,
        verbose: bool = True
    ):
        n_samples = x_train.shape[0]
        best_loss = float('inf')
        wait = 0
        
        self.logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")

        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch loop
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward
                y_pred = self.model.forward(x_batch, training=True)
                
                # Loss & Gradient
                loss = self.loss_fn.compute(y_pred, y_batch)
                grad = self.loss_fn.gradient(y_pred, y_batch)
                
                # Backward
                self.model.backward(grad)
                
                # Update
                self.optimizer.step(self.model.layers)

                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.history['loss'].append(avg_loss)

            # Validation
            val_msg = ""
            if x_val is not None and y_val is not None:
                y_val_pred = self.model.forward(x_val, training=False)
                val_loss = self.loss_fn.compute(y_val_pred, y_val)
                self.history['val_loss'].append(val_loss)
                val_msg = f", Val Loss: {val_loss:.4f}"

                # Early Stopping
                if early_stopping:
                    if val_loss < best_loss - min_delta:
                        best_loss = val_loss
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            if verbose:
                                self.logger.info(f"Epoch {epoch + 1}: Early stopping. Best Val Loss: {best_loss:.4f}")
                            break

            if verbose and (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}{val_msg}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.forward(x, training=False)
