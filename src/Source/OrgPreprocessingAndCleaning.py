
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

# Constants
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
    if data['LotFrontage'].isnull().any():
        predict_missing_values(data, 'LotFrontage', 'LotArea')
    data['GarageYrBlt'].fillna(0, inplace=True)
    data['HasGarage'] = data['GarageYrBlt'].apply(lambda x: 1 if x > 0 else 0)
    columns_to_fill = [
        'TotalBsmtSF', 'BsmtFullBath', 'GarageCars', 'GarageArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtHalfBath', 'MasVnrArea'
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
    else:
        return 'Fall'

def engineer_features(data):
    """Add new features and handle categorical variables."""
    data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']
    data['HasBasement'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']
    data['OverallScore'] = data['OverallQual'] + data['OverallCond']
    data['LotFrontageRatio'] = data['LotFrontage'] / data['LotArea']
    data['SaleSeason'] = data['MoSold'].apply(get_season)
    
    categorical_cols = [
        'MSSubClass', 'Alley', 'MSZoning', 'Street', 'LotShape',
        'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'Foundation', 'Heating', 'CentralAir',
        'Functional', 'GarageType', 'GarageFinish', 'PavedDrive',
        'MiscFeature', 'SaleType', 'SaleCondition', 'SaleSeason',
        'MasVnrType', 'Electrical', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
        'Fence'
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
    if len(sys.argv) > 1:
        train_file_path = sys.argv[1]
        test_file_path = sys.argv[2]
        output_train_file_path = sys.argv[3]
        output_test_file_path = sys.argv[4]
    else:
        train_file_path = TRAIN_FILE_PATH
        test_file_path = TEST_FILE_PATH
        output_train_file_path = OUTPUT_TRAIN_FILE_PATH
        output_test_file_path = OUTPUT_TEST_FILE_PATH

    # Load and preprocess training data
    data = load_and_initial_clean(train_file_path)
    data.drop('Id', axis=1, inplace=True)
    data = fill_missing_values(data)
    data = engineer_features(data)
    add_missing_features(data, ['MSSubClass_150.0'])

    save_preprocessed_data(data, output_train_file_path)

    # Load and preprocess test data
    test_data = load_and_initial_clean(test_file_path)
    test_data = fill_missing_values(test_data)
    test_data = engineer_features(test_data)
    add_missing_features(test_data, [
        'PoolQC_Fa', 'Condition2_RRAe', 'Heating_OthW', 'RoofMatl_Metal',
        'Condition2_RRAn', 'RoofMatl_Roll', 'Electrical_Mix', 'HouseStyle_2.5Fin',
        'Heating_Floor', 'RoofMatl_Membran', 'Condition2_RRNn', 'MiscFeature_TenC',
        'Exterior2nd_Other', 'Exterior1st_Stone', 'Utilities_NoSeWa', 'RoofMatl_ClyTile',
        'GarageQual_Ex', 'Exterior1st_ImStucc'
    ])
    
    save_preprocessed_data(test_data, output_test_file_path)

if __name__ == "__main__":
    main()