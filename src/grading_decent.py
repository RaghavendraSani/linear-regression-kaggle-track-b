import pandas as pd
import numpy as np

train = pd.read_csv("../data/train.csv")

X = train.drop(columns=["Id", "Target"]).values
y = train["Target"].values

n_features = X.shape[1]

weights = np.zeros(n_features)
bias = 0.0

y_pred = X @ weights + bias

error = y_pred - y
mse = np.mean(error**2)

learning_rate = 0.000001
epochs = 1000

n = len(y)

for epoch in range(epochs):
    y_pred = X @ weights + bias
    error = y_pred - y
    mse = np.mean(error**2)

    dw = (2/n) * np.dot(X.T, (y_pred - y))
    db = (2/n) * np.sum(y_pred - y)

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    if (epoch+1)%100 == 0:
        print(f"Epoch: {epoch+1}, mse: {mse}")

print("final weights: ", weights)
print("final bias: ", bias)