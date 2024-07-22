# Housing Price Prediction Model
Final Project for University of Florida CAP 4770 (Intro to Data Science)

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Features](#features)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [License](#license)

## Overview
This project is a take on the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) competition on Kaggle. 
The goal is to predict the final price of each home in Ames, Iowa, using advanced regression techniques.

The repo contains a machine learning pipeline for predicting housing prices using various regression models with the following key features:
- data preprocessing
- feature engineering
- model training
- evaluation of prediction accuracy
- generating predictions for test data

The script processes housing data to prepare it for machine learning models, trains a Gradient Boosting Regressor and a Neural Network, and evaluates the models' performance. 
It also makes predictions on a test dataset and computes the Logarithmic RMSE to assess prediction accuracy.

## Setup

To run the code, you will need to install the following dependencies:

```bash
pip install pandas scikit-learn numpy matplotlib
```

You will also need the following data files:
- [test.csv](data/raw/test.csv)
- [train.csv](data/raw/train.csv)
- [sample_submission.csv](data/raw/sample_submission.csv)


## Features
- Data Preprocessing:
  - Handles missing values using median or predictive imputation
  - Adds new features such as total square footage, house age, and bathroom counts
  - One-hot encodes categorical variables
- Model Training:
  - Trains a Gradient Boosting Regressor with specified hyperparameters
  - Trains a Neural Network using MinMaxScaler and MLPRegressor
- Model Evaluation:
  - Evaluates model performance using K-Fold Cross-Validation
  - Computes Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and compares to a baseline
- Prediction:
  - Generates predictions for test data and saves them for submission


## Model Evaluation
The script evaluates models using the following metrics:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Provides the square root of the MSE, representing the standard deviation of the prediction errors
- **Logarithmic RMSE**: Computes the RMSE on the log-transformed scale to address scale bias in predictions


## Results
The script prints the following results:

- MSE and RMSE for each fold in cross-validation
- Average MSE and RMSE
- RMSE as a percentage of the average sale price
- Comparison of model RMSE to a baseline RMSE


## License
This project is [licensed](LICENSE) under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).

#### Why GPL-3.0?
We chose the GPL-3.0 license to ensure that this project remains open-source and to protect against commercial exploitation. 
The GPL-3.0 license has the following key features:

- **Copyleft**: Any derivative work based on this project must also be distributed under the GPL-3.0 license. This means that improvements and modifications to this code will also be open-source and available to the community.
- **Prevents Proprietary Use**: Commercial entities cannot use this code in proprietary software without also releasing their modifications under the GPL-3.0. This helps to ensure that the software remains free and open for everyone.
