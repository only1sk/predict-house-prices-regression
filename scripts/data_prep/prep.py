# import modules and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer


def fill_na(dataframe):
    """
    Fill missing values in specified columns of a pandas DataFrame.
    
    Parameters:
    dataframe (pd.DataFrame): The pandas DataFrame to be modified.
    
    Returns:
    pd.DataFrame: The modified DataFrame with filled missing values.
    """

    # Filling missing values with specific replacements
    replacements = {
        'PoolQC': 'No pool',
        'MiscFeature': 'None',
        'Alley': 'No alley access',
        'Fence': 'No fence',
        'MasVnrType': 'None',
        'FireplaceQu': 'No Fireplace',
        'GarageType': 'No Garage',
        'BsmtExposure': 'No Basement',
        'BsmtFinType2': 'No Basement',
        'BsmtQual': 'No Basement',
        'BsmtCond': 'No Basement',
        'BsmtFinType1': 'No Basement',
        'GarageFinish': 'No Garage',
        'GarageQual': 'No Garage',
        'GarageCond': 'No Garage'
    }

    for column, replacement in replacements.items():
        dataframe[column] = dataframe[column].fillna(replacement)

    # Replace the NaN value in 'Electrical' with the most common feature value
    most_common = dataframe['Electrical'].value_counts().index[0]
    dataframe['Electrical'] = dataframe['Electrical'].fillna(most_common)

    # Fill 'LotFrontage' with the median
    dataframe['LotFrontage'] = dataframe['LotFrontage'].fillna(dataframe['LotFrontage'].median())

    # Utilising KNN imputer for 'GarageYrBlt' and 'MasVnrArea'
    imputer = KNNImputer(n_neighbors=5)
    imputed_values = imputer.fit_transform(dataframe[['GarageYrBlt', 'MasVnrArea']])

    # Updating the dataframe with imputed values
    dataframe['MasVnrArea'] = pd.DataFrame(imputed_values).iloc[:, 1].astype('float64')
    dataframe['GarageYrBlt'] = pd.DataFrame(imputed_values).iloc[:, 0].astype('int64')

    return dataframe

def column_overview(dataframe, col, bin_size=100):
    """
    Provides statistics on the number of NaN values, unique values and plots the distribution of a specific column in the supplied dataframe object.
    
    Parameters:
    dataframe (pd.DataFrame): The DataFrame to analyze.
    col (str): The column name to analyze.
    bin_size (int): The number of bins for histogram.

    Returns:
    None: This function outputs statistics and plots directly without returning any value.
    """
    # Check for NaN values
    na_values = dataframe[dataframe[col].isna()].shape[0]
    print(f"{col} contains {na_values} NA values\n")

    # Determine cardinality
    unique_values = dataframe[col].nunique()
    print(f"{col} contains {unique_values} unique values\n")
    print(dataframe[col].dtype)

    # Plot distribution
    if dataframe[col].dtype == 'object':
        print("Categorical Data")
        sns.countplot(x=col, data=dataframe)
        plt.title(f"{col} distribution")
        plt.show()
    elif np.issubdtype(dataframe[col].dtype, np.number):
        print("Numerical Data")
        sns.histplot(dataframe[col], bins=bin_size)
        plt.title(f"{col} distribution")
        plt.show()
    else:
        print("Non-numerical or non-categorical data")

def determine_cardinality(dataframe):
    """
    Determine the cardinality of categorical features in the supplied dataframe.

    This function categorizes each categorical feature in the dataframe
    into 'low', 'medium', or 'high' cardinality categories.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame with categorical features.

    Returns:
    tuple: A tuple containing three lists - column names (list of str), count of unique values (list of int),
           and their cardinality categorization (list of str).
    """
    # Only include categorical features
    categorical_features = dataframe.select_dtypes(include=["object"])

    # Categorisation of the number of unique values
    column_names = categorical_features.columns
    unique_values_count = [categorical_features[col].nunique() for col in column_names]
    categorization = [
        "high" if count >= 40 else "medium" if 10 <= count < 40 else "low"
        for count in unique_values_count
    ]
    
    return column_names, unique_values_count, categorization

