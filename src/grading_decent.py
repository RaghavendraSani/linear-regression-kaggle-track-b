import pandas as pd
import numpy as np
np.random.seed(42)

train = pd.read_csv("../data/train.csv")

X = train.drop(columns=["Id", "Target"]).values
y = train["Target"].values

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

split = int(len(X) * 0.8)
print("split index: ", split)

X_train = X[:split]
X_val = X[split:]
y_train = y[:split]
y_val = y[split:]

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

n_features = X.shape[1]

weights = np.zeros(n_features)
bias = 0.0

learning_rate = 0.01
epochs = 1000

n = len(y_train)

for epoch in range(epochs):
    y_pred = X_train @ weights + bias
    error = y_pred - y_train
    mse = np.mean(error**2)

    dw = (2/n) * (X_train.T @ error)
    db = (2/n) * np.sum(error)

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    if (epoch+1)%100 == 0:
        print(f"Epoch: {epoch+1}, mse: {mse}")

val_pred = X_val @ weights + bias
val_error = val_pred - y_val
val_mse = np.mean(val_error**2)
print("\nvalidation MSE: ", val_mse)
print("\nfinal weights: ", weights)
print("\nfinal bias: ", bias)

original_weights = weights / std
original_bias = bias - np.sum((weights * mean) / std)

print("\nOriginal-space weights:")
print(original_weights)

print("\nOriginal-space bias:")
print(original_bias)

test = pd.read_csv("../data/test.csv")
print(test.columns)
X_test = test.drop(columns=["Id"]).values
X_test = (X_test - mean) / std
predictions = X_test @ weights + bias
submission = pd.DataFrame({
    "Id": test["Id"],
    "Expected": predictions
})
submission.to_csv("../submission.csv", index=False)
print("\nSubmission file created successfully!")
print(submission.head())