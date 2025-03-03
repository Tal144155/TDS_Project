import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import chi2_contingency, f_oneway


import warnings
warnings.filterwarnings('ignore')


def column_to_date(df):
    """This function recognize columns that are in the forma of date."""
    date_pattern = r'^(\d{4}-\d{2}-\d{2})|^(\d{2}/\d{2}/\d{4})|^(\d{4}/\d{2}/\d{2})'
    for column in df.columns:
        if df[column].dtype == 'object':
            if df[column].str.match(date_pattern).any():
                try:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    print(f"Converted column '{column}' to datetime.")
                except Exception as e:
                    print(f"Warning: Could not parse column {column} as datetime. {str(e)}")

def read_data(dataset_path, index_col = None):
    """This function checks that the dataset exists in the given path, and read the data using pandas.
    If the data does not exists in the path, print a message to the user and exit."""
    print("- Loading the dataset.")
    if not os.path.exists(dataset_path):
        print(f"Error: The file '{dataset_path}' does not exist. Please check the path and try again.")
        return None
    if index_col:
        df = pd.read_csv(dataset_path, index_col = index_col)
    else:
        df = pd.read_csv(dataset_path)
    column_to_date(df)
    return df

def is_potentially_categorical(column, threshold=0.01):
    """This function determines if an integer column is categorical or numeric. 
    If we have very little unique integer values, the column is probably categorical"""
    unique_values = column.nunique()
    total_values = len(column)
    # check if the percentage of unique values in the column is smaller then the threshold.
    if unique_values / total_values < threshold and unique_values < 20:
        return True
    return False

def get_column_types(df):
    """This function determines the type of each column in our dataset, in order to do smart visualization later.
    Types we recognize: integer, categorical int, float, boolean, string, categorical string, date, object, other."""
    print("- Finding features types in the dataset.")
    column_types = {}
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            if is_potentially_categorical(df[column]):
                column_types[column] = 'categorical_int'
            else:
                column_types[column] = 'integer'
        elif pd.api.types.is_float_dtype(df[column]):
            column_types[column] = 'float'
        elif pd.api.types.is_bool_dtype(df[column]):
            column_types[column] = 'boolean'
        elif pd.api.types.is_string_dtype(df[column]):
            if is_potentially_categorical(df[column]):
                column_types[column] = 'categorical_string'
                df[column] = df[column].astype('category')
            else:
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


def correlation_relations(df, numerical_columns, target_variable, relations, correlation_threshold=0.5):
    # High Correlation Relations (Excluding Target Variable)
    correlations = df[numerical_columns].drop(columns=[target_variable], errors='ignore').corr()
    for i, feature1 in enumerate(correlations.columns):
        for feature2 in correlations.columns[i + 1:]:
            corr_value = correlations.loc[feature1, feature2]
            if abs(corr_value) > correlation_threshold:
                relations.append(
                    {'attributes': [feature1, feature2],
                     'relation_type': 'high_correlation',
                     'details': {'correlation_value': corr_value}})

def correlation_target_value(df, numerical_columns, target_variable, relations, correlation_threshold=0.5):
    # Relations with the Target Variable
    correlations = df[numerical_columns].corr()
    if target_variable in numerical_columns:
        for feature in numerical_columns:
            if feature != target_variable:
                corr_value = correlations.loc[feature, target_variable]
                if abs(corr_value) > correlation_threshold:
                    relations.append({
                        'attributes': [feature, target_variable],
                        'relation_type': 'target_correlation',
                        'details': {'correlation_value': corr_value}
                    })

def categorical_effects(df, categorical_columns, numerical_columns, target_variable, relations, p_value_threshold=0.05):
    if target_variable in numerical_columns:
        for cat_feature in categorical_columns:
            groups = [df[df[cat_feature] == cat][target_variable].dropna() for cat in df[cat_feature].unique()]
            if len(groups) > 1:
                f_stat, p_value = f_oneway(*groups)
                if p_value < p_value_threshold:
                    relations.append({'attributes': [cat_feature, target_variable],'relation_type': 'categorical_effect','details': {'p_value': p_value}})



def find_relations(df, target_variable, dataset_types):
    relations = []
    numerical_columns = [col for col, col_type in dataset_types.items() if col_type in ['integer', 'float']]
    categorical_columns = [col for col, col_type in dataset_types.items() if col_type in ['categorical_int', 'categorical_string']]
    datetime_columns = [col for col, col_type in dataset_types.items() if col_type == 'datetime']

    # Get the relations with high correlation
    correlation_relations(df, numerical_columns, target_variable, relations)

    # Get the relations with the target value
    correlation_target_value(df, numerical_columns, target_variable, relations)

    categorical_effects(df, categorical_columns, numerical_columns, target_variable, relations)

    print(relations)

    return relations



def main():
    dataset_path = "Final Project/Datasets_Testing/dataset_movies.csv"
    # input("Please enter the path to your Dataset: ")
    index_col = "id"
    # input("Please enter the index column: ")
    target_value = "revenue"
    # input("Please enter the name of your target value: ")
    df = read_data(dataset_path)
    if df is None:
        return
    # Understanding the types of columns in the data in order to create better visualizations.
    dataset_types = get_column_types(df)
    # Calling method to get the relations in the data
    find_relations(df, target_value, dataset_types)
    


if __name__ == "__main__":
    main()