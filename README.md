# Linear Regression Kaggle Competition (Track B)

## Overview

This project was developed for the ACM Quest 2026 Linear Regression Competition.

The objective is to predict a continuous target variable using five numerical features. The model is implemented completely from scratch using NumPy and Pandas, following the Track B competition rules.

## Method Used

Normal Equation Linear Regression

The model computes the optimal weight vector using:

w = (XᵀX)⁻¹Xᵀy

A bias column is added manually and predictions are generated through matrix multiplication.

## Project Structure

data/

* train.csv
* test.csv
* sample_submission.csv

src/

* data_loader.py
* eda.py
* metrics.py
* normal_equation.py
* train.py
* generate_submission.py

## Evaluation Metric

Root Mean Squared Error (RMSE)

## Libraries Used

* NumPy
* Pandas

No machine learning frameworks such as scikit-learn, TensorFlow, PyTorch, XGBoost, or LightGBM were used.

## Competition Track

Track B – From Scratch Implementation

## Author

Raghavendra Sani
