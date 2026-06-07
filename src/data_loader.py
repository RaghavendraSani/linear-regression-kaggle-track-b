import pandas as pd

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

print("Train Shape: ", train.shape)
print("Test Shape: ", test.shape)

print("\nTrain Columns: ")
print(train.columns)
print("\nFirst 5 Rows: ")
print(train.head())