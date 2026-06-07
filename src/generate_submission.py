import pandas as pd

from normal_equation import NormalEquationRegression

# Load data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# Training data
X_train = train.drop(columns=["Id", "Target"]).values
y_train = train["Target"].values

# Test data
X_test = test.drop(columns=["Id"]).values

# Train model
model = NormalEquationRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    "Id": test["Id"],
    "Expected": predictions
})

# Save
submission.to_csv("../submissions/submission.csv", index=False)

print("Submission file created successfully!")
print(submission.head())