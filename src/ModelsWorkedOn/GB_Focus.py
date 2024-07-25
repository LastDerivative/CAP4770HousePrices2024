#GB Focus after tuning, with jacks pre-processing and outlier removal
#No tuning and no jack 0.17454
#No tuning with jack .13340
#Tuning with jack .13025
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
#from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

# Constants
TRAIN_FILE_PATH = 'train.csv'
TEST_FILE_PATH = 'test.csv'
OUTPUT_TRAIN_FILE_PATH = 'train_with_features_Gus.csv'
OUTPUT_TEST_FILE_PATH = 'test_with_features_Gus.csv'
SUBMISSION_GB_FILE_PATH = 'predictions_GB.csv'
SUBMISSION_NN_FILE_PATH = 'predictions_NN.csv'
SAMPLE_SUBMISSION_FILE_PATH = 'sample_submission.csv'

# Data Loading and Initial Cleaning
def load_and_initial_clean(filepath):
    """Load the dataset and drop irrelevant columns."""
    return pd.read_csv(filepath)

def fill_missing_values(data):
    """Fill missing numerical values and drop rows for specific cases."""
    data['LotArea'].fillna(data['LotArea'].median(), inplace=True)
    data['HasGarage'] = data['GarageCars'].apply(lambda x: 1 if x > 0 else 0)
    columns_to_fill = [
        'TotalBsmtSF', 'BsmtFullBath', 'GarageCars', 'GarageArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtHalfBath'
    ]
    data[columns_to_fill] = data[columns_to_fill].fillna(0)
    return data

def predict_missing_values(data, target_column, predictor_column):
    """Predict missing values in a target column using a predictor column."""
    non_na_data = data.dropna(subset=[target_column])
    model = LinearRegression()
    model.fit(non_na_data[[predictor_column]], non_na_data[target_column])
    missing_indices = data[target_column].isnull()
    data.loc[missing_indices, target_column] = model.predict(data.loc[missing_indices, [predictor_column]])

# Feature Engineering
def get_season(month):
    """Convert month number to season name."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def engineer_features(data):
    """Add new features and handle categorical variables."""
    data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']
    data['HasBasement'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']
    data['OverallScore'] = data['OverallQual'] + data['OverallCond']
    data['SaleSeason'] = data['MoSold'].apply(get_season)
    
    categorical_cols = [
        'MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
        'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 'CentralAir', 
        'Functional', 'PavedDrive', 'SaleType', 'SaleCondition', 'SaleSeason', 
        'Electrical', 'ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual'
    ]

    data = pd.get_dummies(data, columns=categorical_cols, dummy_na=True)
    return data

def add_missing_features(data, features):
    """Add missing features to the DataFrame."""
    for feature in features:
        data[feature] = 0

# Data Saving
def save_preprocessed_data(data, output_file_path):
    """Save the preprocessed data to a CSV file."""
    data.to_csv(output_file_path, index=False)

# Model Training and Evaluation
def train_gradient_boosting(X, y):
    """Train a Gradient Boosting model."""
    model = GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=4,
            max_features='sqrt',
            random_state=5,
            min_samples_leaf=15,
            min_samples_split=10,
            loss='huber'
    )
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """Evaluate the model using cross-validation."""
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

def align_features(train, test):
    """Align the features of the train and test sets."""
    train_columns = train.columns
    test = test.reindex(columns=train_columns, fill_value=0)
    return train, test

# Main Execution Block
def main():
    # Load and preprocess training data
    data = load_and_initial_clean(TRAIN_FILE_PATH)
    
    # Remove outliers in data
    ids_to_drop = [524, 1299]
    data.drop(data[data['Id'].isin(ids_to_drop)].index, inplace=True)
    
    # Drop id column
    data.drop('Id', axis=1, inplace=True)
    
    # Remove missing data
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    data = data.drop((missing_data[missing_data['Total'] > 1]).index,axis=1)
    
    
    # Fill in missing data and engineer features
    data = fill_missing_values(data)
    data = engineer_features(data)
    add_missing_features(data, ['MSSubClass_150.0'])
    save_preprocessed_data(data, OUTPUT_TRAIN_FILE_PATH)
    
    # Load and preprocess test data
    test_data = load_and_initial_clean(TEST_FILE_PATH)
    
    # Remove missing data also from the test dataset
    test_data = test_data.drop((missing_data[missing_data['Total'] > 1]).index,axis=1)
    
    test_data = fill_missing_values(test_data)
    test_data = engineer_features(test_data)
    add_missing_features(test_data, [
        'Condition2_RRAe', 'Heating_OthW', 'RoofMatl_Metal',
        'Condition2_RRAn', 'RoofMatl_Roll', 'Electrical_Mix', 'HouseStyle_2.5Fin',
        'Heating_Floor', 'RoofMatl_Membran', 'Condition2_RRNn', 
        'Exterior2nd_Other', 'Exterior1st_Stone', 'Utilities_NoSeWa',
        'Exterior1st_ImStucc'
    ])
    save_preprocessed_data(test_data, OUTPUT_TEST_FILE_PATH)
    
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    
    # Train and evaluate Gradient Boosting model
    gb_model = train_gradient_boosting(X, y)
    print("#####################Evaluating Gradient Boosting Model#########################")
    evaluate_model(gb_model, X, y)

    # Predict on the test data
    test_data = pd.read_csv(OUTPUT_TEST_FILE_PATH)
    X_test = test_data.drop('Id', axis=1)

    # Align test features with train features
    X, X_test = align_features(X, X_test)

    gb_model.fit(X, y)
    y_test_pred = gb_model.predict(X_test)

    # Create a DataFrame for submission
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': y_test_pred
    })
    submission.to_csv(SUBMISSION_GB_FILE_PATH, index=False)
    get_logrmse(SUBMISSION_GB_FILE_PATH, SAMPLE_SUBMISSION_FILE_PATH)
    

if __name__ == "__main__":
    main()