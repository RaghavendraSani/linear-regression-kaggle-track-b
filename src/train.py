import pandas as pd

from normal_equation import NormalEquationRegression
from metrics import rmse


df = pd.read_csv("../data/train.csv")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split = int(len(df) * 0.8)

train_df = df.iloc[:split]
val_df = df.iloc[split:]


X_train = train_df.drop(columns=["Id", "Target"]).values
y_train = train_df["Target"].values

X_val = val_df.drop(columns=["Id", "Target"]).values
y_val = val_df["Target"].values


model = NormalEquationRegression()

model.fit(X_train, y_train)

print("Weights:")
print(model.weights)

preds = model.predict(X_val)

score = rmse(y_val, preds)

print("Validation RMSE:", score)

train_preds = model.predict(X_train)

train_rmse = rmse(y_train, train_preds)

print("Train RMSE:", train_rmse)