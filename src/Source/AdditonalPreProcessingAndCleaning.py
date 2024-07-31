import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

TRAIN_FILE_PATH = 'train.csv'
TEST_FILE_PATH = 'test.csv'

OUTPUT_TRAIN_FILE_PATH = 'train_with_features.csv'
OUTPUT_TEST_FILE_PATH = 'test_with_features.csv'

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

# Main Execution Block
def main():

    train_file_path = TRAIN_FILE_PATH
    test_file_path = TEST_FILE_PATH    
    output_train_file_path = OUTPUT_TRAIN_FILE_PATH
    output_test_file_path = OUTPUT_TEST_FILE_PATH

    # Load and preprocess training data
    data = load_and_initial_clean(train_file_path)
    
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
    save_preprocessed_data(data, output_train_file_path)
    
    # Load and preprocess test data
    test_data = load_and_initial_clean(test_file_path)
    
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

    save_preprocessed_data(test_data, output_test_file_path)
    

if __name__ == "__main__":
    main()