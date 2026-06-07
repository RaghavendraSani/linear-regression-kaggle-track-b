import pandas as pd

train = pd.read_csv("../data/train.csv")

split_index = int(len(train)*0.8)

train_data = train[:split_index]
val_data = train[split_index:]

print("Training rows: ", len(train_data))
print("Validation rows: ", len(val_data))