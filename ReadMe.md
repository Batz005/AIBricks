# 🏰 AIMansion

**AIMansion** is a custom, modular deep learning framework designed with a construction philosophy. Just as a mansion is built brick by brick according to a blueprint, your neural networks are constructed using **Bricks** (layers) following a **Blueprint** (model architecture), and realized by a **Builder** (trainer).

## 🧠 Ideology

Deep learning shouldn't be a black box. It's an architectural endeavor.
*   **Bricks**: The fundamental units. Robust, interchangeable, and single-purpose.
*   **Blueprints**: The design. How bricks fit together to form a structure.
*   **Construction**: The process. Taking a design and data to build a functional model.

## 🏗️ Modules

### 🧱 Bricks (`bricks/`)
The building blocks of your model.
*   `DenseBrick`: A fully connected layer.
*   `DropoutBrick`: Randomly drops connections to prevent overfitting.
*   `BatchNormBrick`: Normalizes inputs for stable training.
*   `ActivationBrick`: Applies non-linear functions (ReLU, Linear, etc.).

### 📐 Blueprints (`blueprints/`)
The architectural plans.
*   `SequentialBlueprint`: A linear stack of bricks.

### 👷 Construction (`construction/`)
The tools to build and train.
*   `ModelBuilder`: Manages the training loop, mini-batching, validation, and early stopping.

### 🛠️ Tools
*   `optimizers/`: SGD, Adam (with Gradient Clipping & Decay).
*   `losses/`: MeanSquaredError.
*   `initializers/`:
    *   `He`: Best for ReLU.
    *   `Xavier`: Best for Sigmoid/Tanh.
    *   `Auto`: Automatically selects based on activation.
*   `regularizers/`:
    *   `L1`, `L2`, `ElasticNet`: Weight decay methods to prevent overfitting.
*   `activations/`:
    *   `Relu`, `Linear`, `Sigmoid`, `Tanh`: Non-linear functions for `ActivationBrick`.

## 🚀 Example Usage

Here is how you build a mansion:

```python
import numpy as np
from blueprints.sequential import SequentialBlueprint
from bricks.dense import DenseBrick
from bricks.activation import ActivationBrick
from bricks.dropout import DropoutBrick
from activations.relu import Relu
from activations.linear import Linear
from construction.builder import ModelBuilder
from optimizers.adam import Adam
from losses.MeanSquaredError import MeanSquaredError

# 1. Design the Blueprint
mansion = SequentialBlueprint()
mansion.add(DenseBrick(input_dim=10, output_dim=64))
mansion.add(ActivationBrick(Relu()))
mansion.add(DropoutBrick(rate=0.2))
mansion.add(DenseBrick(input_dim=64, output_dim=1))
mansion.add(ActivationBrick(Linear()))

# 2. Prepare the Materials (Data)
X_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000, 1)

# 3. Hire a Builder
optimizer = Adam(learning_rate=0.001)
loss_fn = MeanSquaredError()
builder = ModelBuilder(mansion, optimizer, loss_fn)

# 4. Build (Train)
builder.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    early_stopping=True,
    patience=5
)

# 5. Inspect (Predict)
predictions = builder.predict(X_train[:5])
print(predictions)
```

## 🕵️‍♀️ Logging & Debugging
AIMansion comes with a built-in logger to help you see what's happening inside.

```python
from utils.logger import set_log_level

# Show training progress (Default)
set_log_level("INFO")

# Show detailed shape information for debugging
set_log_level("DEBUG")
```

## 🧪 Testing
Run the comprehensive test suite to ensure your bricks are solid:
```bash
python3 -m unittest discover tests
```
