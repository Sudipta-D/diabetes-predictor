"""
preprocessing.py module for diabetes prediction project using BRFSS 2015 dataset.

This module provides functions to load data, perform cleaning,
feature transformations, and splitting/scaling for model pipeline.

Features fixed as used in the Streamlit app (21 features).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# Constants for feature ordering and target column
FEATURE_COLS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]
TARGET_COL = 'Diabetes_binary'

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.

    Parameters:
        path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Loaded dataframe with raw data.

    Raises:
        FileNotFoundError: If the CSV file is not found.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load data from {path}: {e}")
    return df

def clean_and_transform(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean and transform the dataframe:
    - Replace 0 values in BMI, MentHlth, PhysHlth with NaN and impute with median.
    - Apply log1p transform to BMI, MentHlth, PhysHlth to reduce skewness.
    - Select and order the fixed features as used in the app.
    - Separate features (X) and target (y).

    Parameters:
        df (pd.DataFrame): Raw dataframe including target and features.

    Returns:
        X (pd.DataFrame): DataFrame containing the 21 features in fixed order.
        y (pd.Series): Series containing the target variable.
    
    Raises:
        KeyError: If expected columns are not in the dataframe.
    """
    # Check required columns
    required_cols = set(FEATURE_COLS + [TARGET_COL])
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in dataframe: {missing}")
    
    # Work on a copy to avoid modifying original df
    data = df.copy()

    # Columns to clean
    cols_to_clean = ['BMI', 'MentHlth', 'PhysHlth']
    # Replace zeros with NaN for specified columns
    data[cols_to_clean] = data[cols_to_clean].replace(0, np.nan)
    # Impute NaN with median of each column
    for col in cols_to_clean:
        median_val = data[col].median()
        data[col].fillna(median_val, inplace=True)
    # Apply log1p transformation
    data[cols_to_clean] = np.log1p(data[cols_to_clean])

    # Select and order features
    X = data[FEATURE_COLS].copy()
    # Extract target
    y = data[TARGET_COL].copy()

    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split features and target into train and test sets.

    Parameters:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of data to be used as test set.
        random_state (int): Seed for random shuffling.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Scale features using StandardScaler (zero mean, unit variance).
    Fit scaler on training data and apply to both train and test sets.

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.

    Returns:
        X_train_scaled (pd.DataFrame): Scaled training features.
        X_test_scaled (pd.DataFrame): Scaled test features.
        scaler (StandardScaler): Fitted scaler object.
    """
    scaler = StandardScaler()
    # Fit on training data
    X_train_scaled_array = scaler.fit_transform(X_train)
    # Transform test data
    X_test_scaled_array = scaler.transform(X_test)

    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled_array, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, index=X_test.index, columns=X_test.columns)

    return X_train_scaled, X_test_scaled, scaler
