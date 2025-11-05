import numpy as np
from models import DenseBrick
from layers import Dense
from activations import Sigmoid
from losses import BinaryCrossEntropy

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = DenseBrick([
    Dense(2, 4, activation=Sigmoid()),
    Dense(4, 1, activation=Sigmoid())
])

model.train(X,y, BinaryCrossEntropy(), epochs=5000, learning_rate = 0.1)

preds = model.predict(X)
print("Predictions:\n", preds)
print("Rounded Predictions:\n", np.round(preds))