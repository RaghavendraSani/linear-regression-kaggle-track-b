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

learning_rate = 0.000001
epochs = 1000

print ("epochs: ",epochs)
print("Learning rate: ",learning_rate)

n = len(y)
dw = (2/n) * np.dot(X.T, (y_pred - y))
db = (2/n) * np.sum(y_pred - y)

print("dw: ",dw)
print("db: ",db)

print("weights before: ",weights)
print("bias brefore: ",bias)

weights = weights - learning_rate * dw
bias = bias - learning_rate * db

print("weights after: ",weights)
print("bias after: ",bias)

y_new_pred = X @ weights + bias
new_error = y_new_pred - y
new_sq_error = np.square(new_error)
new_mse = np.mean(new_sq_error)

print("old mse: ",mse)
print("new mse: ",new_mse)



