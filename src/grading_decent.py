import pandas as pd
import numpy as np

train = pd.read_csv("../data/train.csv")

X = train.drop(columns=["Id", "Target"]).values
y = train["Target"].values

print("x shape: ",X.shape)
print("y shape: ",y.shape)

n_features = X.shape[1]

weights = np.zeros(n_features)
bias = 0.0

print("weights: ",weights)
print("bias: ",bias)

y_pred = X @ weights + bias
print("First 10 predictions: ",y_pred[:10])

error = y_pred - y
sq_error = np.square(error)
mse = np.mean(sq_error)

print("Mean squared error: ",mse)

learning_rate = 0.0001
epochs = 1000

print ("epochs: ",epochs)
print("Learning rate: ",learning_rate)



