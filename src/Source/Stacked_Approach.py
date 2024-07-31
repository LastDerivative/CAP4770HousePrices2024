#attempting stacked with NN, no outlier removal and no additonal pre-processing 0.12512
#pre processing and outliers: 0.12942

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neural_network import MLPRegressor
import datetime
import sys


TRAIN_PROC_FILE_PATH = 'train_with_features.csv'
TEST_PROC_FILE_PATH = 'test_with_features.csv'
PREDICTIONS_STACKED_FILE_PATH = 'predictions_Stacked.csv'
SAMPLE_SUBMISSION_FILE_PATH = 'sample_submission.csv'

# Data Loading and Initial Cleaning
def load_data(filepath):
    """Load the dataset and drop irrelevant columns."""
    return pd.read_csv(filepath)

# Data Saving
def save_preprocessed_data(data, output_file_path):
    """Save the preprocessed data to a CSV file."""
    data.to_csv(output_file_path, index=False)


def evaluate_model(model, X, y):
    """Evaluate the model using cross-validation."""
    print("#####################Evaluating Stacked Model#########################")
    start_time = datetime.datetime.now()

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(mse_scores)  # Convert MSE to RMSE

    print("MSE scores for each fold:", mse_scores)
    print("RMSE scores for each fold:", rmse_scores)
    print("Average MSE:", np.mean(mse_scores))
    print("Average RMSE:", np.mean(rmse_scores))

    average_price = y.mean()
    rmse_percentage = (rmse_scores.mean() / average_price) * 100
    print(f"Average Sale Price: ${average_price:.2f}")
    print(f"Average RMSE: ${rmse_scores.mean():.2f}")
    print(f"RMSE as Percentage of Average Sale Price: {rmse_percentage:.2f}%")

    baseline_rmse = np.sqrt(mean_squared_error(y, [y.mean()] * len(y)))
    print(f"Baseline RMSE: ${baseline_rmse:.2f}")
    if rmse_scores.mean() < baseline_rmse:
        print("Model RMSE is better than baseline.")
    else:
        print("Model RMSE is not better than baseline.")
    
        # End time
    end_time = datetime.datetime.now()
    # Calculate the duration
    duration = end_time - start_time
    print(f"Model Evaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Evaluation finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Evaluation execution time: {duration}")

# Logarithmic RMSE Calculation
def get_logrmse(predictions_path, sample_path):
    """Calculate logarithmic RMSE."""
    predictions_df = pd.read_csv(predictions_path)
    actual_df = pd.read_csv(sample_path)

    predictions_df.sort_values('Id', inplace=True)
    actual_df.sort_values('Id', inplace=True)

    predictions_ids = set(predictions_df['Id'])
    actual_ids = set(actual_df['Id'])
    mismatched_ids = actual_ids - predictions_ids

    if mismatched_ids:
        warnings.warn(f"Mismatched IDs found: {mismatched_ids}. Please correct.")
    else:
        actual_df.sort_values('Id', inplace=True)
        log_predictions = np.log(predictions_df['SalePrice'] + 1)
        log_actual = np.log(actual_df['SalePrice'] + 1)
        mse = mean_squared_error(log_actual, log_predictions)
        rmse = np.sqrt(mse)
        print(f'Logarithmic RMSE: {rmse}')

def train_stacked_model(X, y):

    print("#####################Training Stacked Model#########################")

    # Define base models
    base_models = [
        ('gb', GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=4,
            max_features='sqrt',
            random_state=42,
            min_samples_leaf=15,
            min_samples_split=10,
            loss='huber',
        )),
        ('mlp', make_pipeline(
            MinMaxScaler(),
            MLPRegressor(
            hidden_layer_sizes=(128, 64, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate_init=0.000093,
            max_iter=10000,
            random_state=42
            )
        ))
    ]
    # Define meta-model
    meta_model = Ridge(alpha=1.0)
    
    # Create the stacking regressor with scaling
    stacked_model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('stacking', StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        ))
    ])

    start_time = datetime.datetime.now()
    stacked_model.fit(X, y)
    end_time = datetime.datetime.now()

    # Calculate the duration
    duration = end_time - start_time
    print(f"Model Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Training finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Training execution time: {duration}")

    return stacked_model

def align_features(train, test):
    """Align the features of the train and test sets."""
    train_columns = train.columns
    test = test.reindex(columns=train_columns, fill_value=0)
    return train, test

# Main Execution Block
def main():
    train_proc_file_path = TRAIN_PROC_FILE_PATH
    test_proc_file_path = TEST_PROC_FILE_PATH
    predictions_stacked_file_path = PREDICTIONS_STACKED_FILE_PATH   
    evaluate_stacked = False
    if len(sys.argv) > 1:#File is being run from pipeline
        sample_submission_file_path = sys.argv[4]
        if(sys.argv[5].lower() == 'true'):
            evaluate_stacked = True
    else:
        sample_submission_file_path = SAMPLE_SUBMISSION_FILE_PATH    

    # Load and preprocess training data
    data = load_data(train_proc_file_path)
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    if(evaluate_stacked):
        print("#####################Training Stacked Model#########################")

    # Fit and evaluate the Neural Network model
    stacked_model = train_stacked_model(X, y)

    if(evaluate_stacked):
        evaluate_model(stacked_model, X, y)

    # Load test data
    test_data = load_data(test_proc_file_path)
    X_test = test_data.drop('Id', axis=1)

    # Align test features with train features
    X, X_test = align_features(X, X_test)

    # Predict on the test data
    y_test_pred = stacked_model.predict(X_test)

    # Create a DataFrame for submission
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': y_test_pred
    })
    submission.to_csv(predictions_stacked_file_path, index=False)
    get_logrmse(predictions_stacked_file_path, sample_submission_file_path)

if __name__ == "__main__":
    main()