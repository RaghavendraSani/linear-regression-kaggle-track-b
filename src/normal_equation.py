import numpy as np


class NormalEquationRegression:

    def fit(self, X, y):

        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):

        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        return X @ self.weights