import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np

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

def read_data(dataset_path, index_col):
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


def correlation_heatmap_visualize(df, dataset_types, target_variable, output_folder):
    # Visualize heatmap correlation between numerical features and the target value.
    # Might not be a good visualization, depends on the number of features that are numerical.
    numerical_columns = [col for col, col_type in dataset_types.items() if col_type in ['integer', 'float']]
    if target_variable in numerical_columns:
        correlation_matrix = df[numerical_columns].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.title('Correlation Matrix (Numerical Features)')
        plt.savefig(os.path.join(output_folder, 'correlation_matrix.png'))
        plt.close()

def high_correlation_features(df, dataset_types, target_variable, output_folder, correlation_threshold=0.5):
    # Create scatter plots for highly correlated features
    numerical_columns = [col for col, col_type in dataset_types.items() if col_type in ['integer', 'float']]
    correlations = df[numerical_columns].corr()

    # Iterate over all possible pairs of features
    for i, feature1 in enumerate(numerical_columns):
        for feature2 in numerical_columns[i + 1:]:
            corr_value = correlations.loc[feature1, feature2]
            if abs(corr_value) > correlation_threshold:
                plt.figure(figsize=(8, 6))
                
                # Scatter plot with regression line
                sns.regplot(x=df[feature1], y=df[feature2], scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'color': 'red'})
                
                plt.xlabel(feature1)
                plt.ylabel(feature2)
                plt.title(f'Correlation: {corr_value:.2f} | {feature1} vs {feature2}')
                plt.legend()
                plt.grid(True)

                # Save the plot with a descriptive filename
                filename = f'scatter_{feature1}_vs_{feature2}.png'
                plt.savefig(os.path.join(output_folder, filename))
                plt.close()


def generate_visualizations(dataset_path, target_variable, output_folder, index_col = None):
    # Trying to load the dataset, if it does not work exist the process.
    df = read_data(dataset_path, index_col)
    if df is None:
        return
    # Create output directory if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Understanding the types of columns in the data in order to create better visualizations.
    dataset_types = get_column_types(df)

    correlation_heatmap_visualize(df, dataset_types, target_variable, output_folder)
    high_correlation_features(df, dataset_types, target_variable, output_folder)
    
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
    
    dataset_path = "Final Project/Datasets_Testing/dataset_movies.csv"
    # input("Please enter the path to your Dataset: ")
    index_col = "id"
    # input("Please enter the index column: ")
    target_value = "revenue"
    # input("Please enter the name of your target value: ")
    print("""
    =========================================
               Beginning the process
    =========================================
    """)
    generate_visualizations(dataset_path, target_value, "output")

    


if __name__ == "__main__":
    main()
