import os
import pandas as pd
from pandas import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import re

import warnings
warnings.filterwarnings('ignore')


def column_to_date(df):
    # Check for potential date columns using regex
    date_pattern = r'^(\d{4}-\d{2}-\d{2})|^(\d{2}/\d{2}/\d{4})|^(\d{4}/\d{2}/\d{2})'
    for column in df.columns:
        if df[column].dtype == 'object':
            if df[column].str.match(date_pattern).any():
                try:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    print(f"Converted column '{column}' to datetime.")
                except Exception as e:
                    print(f"Warning: Could not parse column {column} as datetime. {str(e)}")

def read_data(dataset_path):
    """This function checks that the dataset exists in the given path, and read the data using pandas.
    If the data does not exists in the path, print a message to the user and exit."""
    print("- Loading the dataset.")
    if not os.path.exists(dataset_path):
        print(f"Error: The file '{dataset_path}' does not exist. Please check the path and try again.")
        return None
    df = pd.read_csv(dataset_path)
    column_to_date(df)
    return df

def is_potentially_categorical(column, threshold=0.02):
    """This function determines if an integer column is categorical or numeric. 
    If we have very little unique integer values, the column is probably categorical"""
    unique_values = column.nunique()
    total_values = len(column)
    # check if the percentage of unique values in the column is smaller then the threshold.
    if unique_values / total_values < threshold:
        return True
    return False

def get_column_types(df):
    """This function determines the type of each column in our dataset,
    in order to do smart visualization later"""
    column_types = {}
    for column in df.columns:
        if isinstance(df[column].dtype, pd.CategoricalDtype):
            column_types[column] = 'categorical'
        elif pd.api.types.is_integer_dtype(df[column]):
            if is_potentially_categorical(df[column]):
                column_types[column] = 'categorical_int'
            else:
                column_types[column] = 'integer'
        elif pd.api.types.is_float_dtype(df[column]):
            column_types[column] = 'float'
        elif pd.api.types.is_bool_dtype(df[column]):
            column_types[column] = 'boolean'
        elif pd.api.types.is_string_dtype(df[column]):
            column_types[column] = 'string'
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            column_types[column] = 'datetime'
        elif pd.api.types.is_timedelta64_dtype(df[column]):
            column_types[column] = 'timedelta'
        elif pd.api.types.is_object_dtype(df[column]):
            column_types[column] = 'object'
        else:
            column_types[column] = 'other'
    return column_types


def generate_visualizations(dataset_path: str, target_variable: str, output_folder: str = 'visualizations'):
    # Trying to load the dataset, if it does not work exist the process.
    df = read_data(dataset_path)
    if df is None:
        return
    # Create output directory if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    dataset_types = get_column_types(df)

    print(dataset_types)
    
    print("""
    =========================================
          Process ended. Results saved.
    =========================================
    """)




def main():
    print("""
    =========================================
            Automated Visualization Tool
    =========================================
    Hi there! Welcome to our automated visualization generator.
    Please follow the instructions below to generate insightful visualizations
    from your dataset automatically.
    =========================================
    """)
    
    dataset_path = "Final Project/Datasets_Testing/AB_NYC_2019.csv"
    # input("Please enter the path to your Dataset: ")
    target_value = "hello"
    # input("Please enter the name of your target value: ")
    print("""
    =========================================
               Beginning the process
    =========================================
    """)
    generate_visualizations(dataset_path, target_value, "output")

    


if __name__ == "__main__":
    main()
