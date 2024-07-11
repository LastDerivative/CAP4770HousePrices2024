import pandas as pd

file_path = 'train.csv'
data = pd.read_csv(file_path)

# Replaces NA with null
data = data.fillna('')

data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']

data['HouseAge'] = data['YrSold'] - data['YearBuilt']

data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']

data['HasBasement'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

data['TotalBath'] = data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])

# Overall Quality Score
data['OverallScore'] = data['OverallQual'] + data['OverallCond']

# Lot Frontage to Lot Area Ratio
data['LotFrontageRatio'] = data['LotFrontage'] / data['LotArea']

# Simplify Quality and Condition Ratings
quality_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
data['ExterQual'] = data['ExterQual'].map(quality_map)
data['BsmtQual'] = data['BsmtQual'].map(quality_map)

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

data['SaleSeason'] = data['MoSold'].apply(get_season)

# Price per SF
data['PricePerSF'] = data['SalePrice'] / data['TotalSF']

output_file_path_features = 'train_with_features.csv'
data.to_csv(output_file_path_features, index=False)