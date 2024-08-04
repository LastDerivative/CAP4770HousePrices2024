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
pip install pandas scikit-learn numpy matplotlib argparse datetime
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
  - Trains a Stacked model that combines both the Gradient Boosting and Neural Networks
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
The code outputs the following results:
- Predictions made by models ran in a format submittable to Kaggle
- MSE and RMSE for each fold in cross-validation
- Average MSE and RMSE
- RMSE as a percentage of the average sale price
- Comparison of model RMSE to a baseline RMSE

```
MSE scores for each fold: [6.63152554e+08 4.91717313e+08 4.91675384e+08 7.76904957e+08 6.05079971e+08]
RMSE scores for each fold: [25751.74855413 22174.6998479  22173.75438609 27873.01484767 24598.37333298]
Average MSE: 605706035.5682492
Average RMSE: 24514.31819375379
Average Sale Price: $180932.92
Average RMSE: $24514.32
RMSE as Percentage of Average Sale Price: 13.55%
Baseline RMSE: $79467.79
Model RMSE is better than baseline.
Model Evaluation started at: 2024-07-30 22:05:50
Model Evaluation finished at: 2024-07-30 22:13:17
Total Evaluation execution time: 0:07:26.727285
Logarithmic RMSE: 0.4167785842875396
```

\
Able to specify which models to run by specifying with the following flags
```
--run_gb --run_nn --run_stacked --run_all
```

- *FullPipeline.py `--run_nn` will only run the nn model with evaluation*
- *By default, evaluation of the stacked model is not run since total run time ~50 minutes*
- *By default, evaluation of the Neural Model is not run since total run time ~50 minutes*

\
Able to specify stacked Evaluation when running the stacked model or all models with `--evaluate_stacked`. For example:
```
FullPipeline.py --run_all --evaluate_stacked or FullPipeline.py --run_stacked --evaluate_stacked
```

*`FullPipeline.py` by default runs the GB model and the Neural Network Model with evaluation on both. Total run time: ~10 minutes*

\
**Results of stacked Evaluation with outlier removal and aggressive preprocessing:**
```
Total Time: 50 minutes

Total Training execution time: 0:10:24.215602
Total Evaluation execution time: 0:40:07.320574

MSE scores for each fold: [4.57695389e+08 4.78728189e+08 4.27827539e+08 6.84540074e+08 4.58626083e+08]
RMSE scores for each fold: [21393.8165989  21879.85806529 20683.99232721 26163.71675458 21415.55703462]
Average MSE: 501483454.7467955
Average RMSE: 22307.38815612143
Average Sale Price: $180932.92
Average RMSE: $22307.39
RMSE as Percentage of Average Sale Price: 12.33%
Baseline RMSE: $79467.79
Model RMSE is better than baseline.

KAGGLE: 0.12942
```


## License
This project is [licensed](LICENSE) under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).

#### Why GPL-3.0?
We chose the GPL-3.0 license to ensure that this project remains open-source and to protect against commercial exploitation. 
The GPL-3.0 license has the following key features:

- **Copyleft**: Any derivative work based on this project must also be distributed under the GPL-3.0 license. This means that improvements and modifications to this code will also be open-source and available to the community.
- **Prevents Proprietary Use**: Commercial entities cannot use this code in proprietary software without also releasing their modifications under the GPL-3.0. This helps to ensure that the software remains free and open for everyone.
