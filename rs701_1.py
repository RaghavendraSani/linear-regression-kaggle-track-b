import numpy as np
import pandas as pd


def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))


class NormalEquationRegression:

    def fit(self, X, y):

        # adding a bias column
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))

        # normal equation
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):

        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))

        return X @ self.w


if __name__ == "__main__":

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X_train = train_df.drop(columns=["Id", "Target"]).values
    y_train = train_df["Target"].values

    X_test = test_df.drop(columns=["Id"]).values

    model = NormalEquationRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "Expected": pred
    })

    submission.to_csv("submission.csv", index=False)

    print("Submission file created successfully!")