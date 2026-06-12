# Linear Regression from Scratch using Gradient Descent

## Overview

This project was developed as part of the ACM ML Quest 2026 Linear Regression Challenge (Track B).

The objective was to implement Linear Regression completely from scratch without using any pre-built machine learning models or `.fit()` methods.

The model is trained using Gradient Descent and predicts the target values for unseen test data.

---

## Features

* Linear Regression implemented from scratch
* Manual Mean Squared Error (MSE) calculation
* Manual Gradient Descent optimization
* Feature Standardization (Z-score Scaling)
* Train/Validation Split
* Validation Performance Evaluation
* Reproducible Results using Random Seed
* Kaggle Submission File Generation

---

## Dataset

### Training Data

* 1600 samples
* 5 input features
* 1 target variable

### Test Data

* 400 samples
* 5 input features

Features:

* Feature_1
* Feature_2
* Feature_3
* Feature_4
* Feature_5

Target:

* Target

---

## Methodology

### 1. Data Loading

The training dataset is loaded using Pandas and separated into:

* Features (X)
* Target (y)

### 2. Data Shuffling

The dataset is randomly shuffled before splitting.

```python
np.random.seed(42)
indices = np.random.permutation(len(X))
```

### 3. Train/Validation Split

The dataset is divided into:

* 80% Training Data
* 20% Validation Data

### 4. Feature Scaling

Features are standardized using:

x_scaled = (x - mean) / std

Mean and standard deviation are computed only on the training set and reused for validation and test data to prevent data leakage.

### 5. Model Initialization

Weights and bias are initialized as:

```python
weights = np.zeros(n_features)
bias = 0.0
```

### 6. Prediction Function

Predictions are generated using:

y_pred = X @ weights + bias

### 7. Loss Function

Mean Squared Error (MSE):

MSE = (1/n) * Σ(y_pred - y)^2

### 8. Gradient Computation

Weight gradients:

dw = (2/n) * Xᵀ(y_pred - y)

Bias gradient:

db = (2/n) * Σ(y_pred - y)

### 9. Gradient Descent Update

```python
weights = weights - learning_rate * dw
bias = bias - learning_rate * db
```

This process is repeated for multiple epochs until convergence.

---

## Hyperparameters

```text
Learning Rate : 0.01
Epochs        : 1000
Random Seed   : 42
```

---

## Results

Training MSE:

```text
0.9672
```

Validation MSE:

```text
1.0062
```

Recovered coefficients in original feature space:

```text
Feature_1 :  3.499
Feature_2 : -2.001
Feature_3 : 12.106
Feature_4 :  0.0498
Feature_5 : -50.050
Bias      :  0.038
```

These values closely match the analytical solution obtained using the Normal Equation, validating the correctness of the Gradient Descent implementation.

---

## Submission Generation

After training:

1. Test features are loaded.
2. Training mean and standard deviation are reused for scaling.
3. Predictions are generated.
4. A submission file is created in the required format:

```text
Id,Expected
1,...
2,...
3,...
```

---

## Technologies Used

* Python
* NumPy
* Pandas

---

## Author

Raghavendra Sani

ACM ML Quest 2026 – Track B

Linear Regression implemented from scratch using Gradient Descent.
