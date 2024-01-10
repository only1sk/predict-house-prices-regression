# import modules and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



def plot_and_transform_skewness_all_cols(dataframe, column):
    """
    Plots the distribution of a specified column in a dataframe, applies log or exponential transformation based on skewness,
    and then plots the transformed distribution.
    
    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    column (str): The column to be transformed.

    Returns:
    - pd.DataFrame: DataFrame with transformed columns.
    """
    # Plot original distribution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(dataframe[column], kde=True)
    plt.title(f'Original Distribution of {column}')

    # Check skewness and apply appropriate transformation
    skewness = dataframe[column].skew()
    if skewness > 0.5:
        # Apply log transformation for right-skewed data
        transformed = np.log1p(dataframe[column])
        transformation_type = 'Log'
    elif skewness < -0.5:
        # Apply exponential transformation for left-skewed data
        transformed = np.expm1(dataframe[column])
        transformation_type = 'Exponential'
    else:
        transformed = dataframe[column]
        transformation_type = 'No Transformation'

    # Plot transformed distribution
    plt.subplot(1, 2, 2)
    sns.histplot(transformed, kde=True)
    plt.title(f'{transformation_type} Transformed Distribution of {column}')

    return transformed

def plot_and_transform_skewness_specific_cols(dataframe, exclude_columns):
    """
    Plots distributions and applies transformations to columns in a DataFrame based on their skewness,
    excluding specified columns.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - exclude_columns (list of str): Column names to exclude from transformation.

    Returns:
    Tuple[pd.DataFrame, dict]: A tuple containing the DataFrame with transformed columns and a dictionary 
    indicating the transformation type for each column.
    """
    transformed_df = dataframe.copy()
    transformations = {}

    for column in dataframe.columns:
        if column not in exclude_columns:
            plt.figure(figsize=(12, 4))

            # Plot original distribution
            plt.subplot(1, 2, 1)
            sns.histplot(dataframe[column], kde=True)
            plt.title(f'Original Distribution of {column}')

            skewness = dataframe[column].skew()

            # Apply transformation based on skewness
            if skewness > 0.5:
                transformed_df[column] = np.log1p(dataframe[column])
                transformation_type = 'Log'
            elif skewness < -0.5:
                transformed_df[column] = np.expm1(dataframe[column])
                transformation_type = 'Exponential'
            else:
                transformation_type = 'No Transformation'

            transformations[column] = transformation_type

            # Plot transformed distribution
            plt.subplot(1, 2, 2)
            sns.histplot(transformed_df[column], kde=True)
            plt.title(f'{transformation_type} Transformed Distribution of {column}')
            plt.show()

    return transformed_df, transformations


def plot_square_root_transformation(dataframe, columns):
    """
    This function plots the original and square root transformed distributions of the specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    columns (list of str): The columns to be transformed.

    Returns:
    pd.DataFrame: A dataframe with the transformed columns.
    """
    transformed_dataframe = dataframe.copy()
    small_constant = 1  # Small constant added to avoid sqrt(0) issues

    for column in columns:
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        # Plot original distribution
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(dataframe[column], kde=True)
        plt.title(f'Original Distribution of {column}')

        # Apply square root transformation
        transformed = np.sqrt(dataframe[column] + small_constant)
        transformed_dataframe[column] = transformed  # Update transformed dataframe

        # Plot transformed distribution
        plt.subplot(1, 2, 2)
        sns.histplot(transformed, kde=True)
        plt.title(f'Square Root Transformed Distribution of {column}')
        plt.show()

    return transformed_dataframe

def scale_numerical_features(numerical_df):
    """
    Scale numerical features in the dataframe and return the scaler.
    
    Parameters:
    numerical_df (pd.DataFrame): The DataFrame containing the numerical data.

    Returns:
    Tuple[pd.DataFrame, StandardScaler]: A tuple containing the dataframe with the scaled features and the scaler object.
    """
    # Initialise the scaler
    scaler = StandardScaler()
    
    # Fit the scaler to the data
    scaler.fit(numerical_df)

    # Transform the data using the fitted scaler
    scaled_data = scaler.transform(numerical_df)
    
    return scaled_data, scaler

def one_hot_encode_dataframe(dataframe):
    """
    Applies one-hot encoding to all categorical columns in a given dataframe.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be transformed.

    Returns:
    pd.DataFrame: A new DataFrame with one-hot encoded columns.
    """
    # Identify categorical columns (assuming columns with 'object' datatype are categorical)
    categorical_cols = dataframe.select_dtypes(include=['object']).columns

    # Initialize OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False, drop='first')

    # Apply OneHotEncoder to categorical columns
    onehot_encoded = onehot_encoder.fit_transform(dataframe[categorical_cols])

    # Create a DataFrame with the one-hot encoded columns
    onehot_encoded_df = pd.DataFrame(
        onehot_encoded, 
        columns=onehot_encoder.get_feature_names_out(categorical_cols)
    )

    # Drop the original categorical columns and concatenate the new one-hot encoded columns
    dataframe = dataframe.drop(columns=categorical_cols).reset_index(drop=True)
    onehot_encoded_df = onehot_encoded_df.reset_index(drop=True)
    transformed_dataframe = pd.concat([dataframe, onehot_encoded_df], axis=1)

    return transformed_dataframe

def separate_variables_for_model_dev(df, label='SalePrice'):
    """
    Separates a pandas DataFrame into features and target variable.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to separate features and target.
    label (str): The name of the column to be used as the target variable. Defaults to 'SalePrice'.

    Returns:
    Tuple[pd.DataFrame, pd.Series]: A tuple containing two elements, the first being a DataFrame with features (X) 
    and the second a Series with the target variable (y).
    """
    # Drop the target column from the DataFrame to create the features set
    X = df.drop(columns=label)

    # Extract the target variable into its own DataFrame
    y = df[label]

    return X, y

