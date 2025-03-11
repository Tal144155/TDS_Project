import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import chi2_contingency, f_oneway
from sklearn.metrics import mutual_info_score
from scipy import stats


import warnings
warnings.filterwarnings('ignore')

TOP_N_RELATIONS = 10

def column_to_date(df):
    """This function recognize columns that are in the forma of date. if it is date, it transforms it"""
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
    # converting date columns to date
    column_to_date(df)
    # dropping null lines (we will not work with them)
    df.dropna(inplace=True)
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
    # go over all possible types, and if it is a match, apply it.
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
    # High Correlation Relations (Excluding Target Variable), checking with Pearson correlation coefficient with a threshold
    print("- Checking for correlation.")
    correlations = df[numerical_columns].drop(columns=[target_variable], errors='ignore').corr()
    for i, feature1 in enumerate(correlations.columns):
        for feature2 in correlations.columns[i + 1:]:
            corr_value = correlations.loc[feature1, feature2]
            # if the correlation exceeds the threshold, add it to the relations
            if abs(corr_value) > correlation_threshold:
                relations.append(
                    {'attributes': [feature1, feature2],
                     'relation_type': 'high_correlation',
                     'details': {'correlation_value': corr_value}})

def correlation_target_value(df, numerical_columns, target_variable, relations, correlation_threshold=0.5):
    """Relations with the Target Variable, does what the regular correlation does but this time with the target variable
    We decided to separate them because this connection might be more interesting to visualize."""
    print("- Checking for correlation with the target variable.")
    correlations = df[numerical_columns].corr()
    if target_variable in numerical_columns:
        for feature in numerical_columns:
            if feature != target_variable:
                corr_value = correlations.loc[feature, target_variable]
                # if the correlation exceeds the threshold, add it to the relations
                if abs(corr_value) > correlation_threshold:
                    relations.append({
                        'attributes': [feature, target_variable],
                        'relation_type': 'target_correlation',
                        'details': {'correlation_value': corr_value}
                    })

def categorical_effects(df, categorical_columns, numerical_columns, target_variable, relations, p_value_threshold=0.05):
    """Relations between categorical feature and numeric feature. for each category, finding the different attributes,
    and running ANOVA test (one way) between them. if the results is small then the p-value determined, the relation is added."""
    print("- Checking for categorical effect.")
    temp_relations = []
    if target_variable in numerical_columns:
        for cat_feature in categorical_columns:
            # finding all attributes in a category
            groups = [df[df[cat_feature] == cat][target_variable].dropna() for cat in df[cat_feature].unique()]
            if len(groups) > 1:
                # running anova test
                f_stat, p_value = f_oneway(*groups)
                # check if the result is lower then the threshold
                if p_value < p_value_threshold:
                    temp_relations.append(
                        {'attributes': [cat_feature, target_variable],
                         'relation_type': 'categorical_effect',
                         'details': {'p_value': p_value}})
    temp_relations.sort(key=lambda x: x['details']['p_value'])
    relations.extend(temp_relations[:TOP_N_RELATIONS])

def chi_squared_relationship(df, categorical_columns, relations, p_value_threshold=0.05):
    """Relations between two categorical features. checking for their chi square result, if it is lower then
    the predefined threshold, add it to the relations."""
    print("- Checking for chi square relation.")
    temp_relations = []
    for i, feature1 in enumerate(categorical_columns):
        for feature2 in categorical_columns[i + 1:]:
            # create a contingency table for the two categorical features
            contingency_table = pd.crosstab(df[feature1], df[feature2])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            # if the p-value is less than the threshold, it indicates a significant relationship
            if p < p_value_threshold:
                temp_relations.append(
                    {'attributes': [feature1, feature2],
                     'relation_type': 'chi_squared',
                     'details': {'p_value': p}})
    temp_relations.sort(key=lambda x: x['details']['p_value'])
    relations.extend(temp_relations[:TOP_N_RELATIONS])

def date_numerical_relationship(df, date_columns, numerical_columns, relations, correlation_threshold=0.35):
    """Relations between date feature and numeric feature. we transform the date to a time ordinal, and then
    check for correlation between the feature and the new date feature. if the correlation is higher 
    then a predefined threshold, add it to the relations."""
    print("- Checking for date with numerical variables.")
    temp_relations = []
    for date_col in date_columns:
        # Safely convert to datetime and drop NaT values
        valid_dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if valid_dates.empty:
            continue
        # adding new feature for time ordinal
        df['time_ordinal'] = valid_dates.map(pd.Timestamp.toordinal)
        for num_feature in numerical_columns:
            # Only use rows where the date is valid
            valid_data = df.loc[valid_dates.index, num_feature].dropna()
            if not valid_data.empty:
                corr_value = df.loc[valid_data.index, 'time_ordinal'].corr(valid_data)
                # if the correlation is higher then the threshold
                if abs(corr_value) > correlation_threshold:
                    temp_relations.append(
                        {'attributes': [date_col, num_feature],
                         'relation_type': 'date_numerical_trend',
                         'details': {'correlation_value': corr_value}}
                    )
    # add only the 10 most noticeable relations.
    temp_relations.sort(key=lambda x: abs(x['details']['correlation_value']), reverse=True)
    relations.extend(temp_relations[:TOP_N_RELATIONS])

def date_categorical_relationship(df, date_columns, categorical_columns, relations, p_value_threshold=0.05):
    """Relations between date feature and categorical feature. we transform the date to a time ordinal, and then
    check for chi 2 test between the feature and the new date feature. if the p-value is lower 
    then a predefined threshold, add it to the relations."""
    print("- Checking for date with categorical variable.")
    temp_relations = []
    for date_col in date_columns:
        # transfer date to period of months
        df['date_period'] = pd.to_datetime(df[date_col]).dt.to_period('M')
        for cat_feature in categorical_columns:
            contingency_table = pd.crosstab(df['date_period'], df[cat_feature])
            # check for chi 2 test
            chi2, p, _, _ = chi2_contingency(contingency_table)
            if p < p_value_threshold:
                temp_relations.append(
                    {'attributes': [date_col, cat_feature],
                     'relation_type': 'date_categorical_distribution',
                     'details': {'p_value': p}}
                )
    # add only the 10 most noticeable relations.
    temp_relations.sort(key=lambda x: x['details']['p_value'])
    relations.extend(temp_relations[:TOP_N_RELATIONS])

def non_linear_relationships(df, numerical_columns, relations, threshold=0.5):
    """Relation between 2 numerical features. using Mutual Information (MI).
    If the MI score between two numerical features exceeds the given threshold, 
    the relationship is recorded."""
    print("- Checking for non linear relation.")
    for col1 in numerical_columns:
        for col2 in numerical_columns:
            if col1 != col2:
                # compute Mutual Information (MI) score using discretized bins
                mi = mutual_info_score(
                    pd.qcut(df[col1], 10, duplicates='drop', labels=False), 
                    pd.qcut(df[col2], 10, duplicates='drop', labels=False)
                )
                # if the MI score is greater than the threshold, consider it a significant relation
                if mi > threshold:
                    relations.append({
                        'attributes': [col1, col2],
                        'relation_type': 'non_linear',
                        'details': {'mutual_information': mi}
                    })

def feature_importance_relations(df, numerical_columns, target_variable, relations, top_n=5):
    """CIdentifies the most important numerical features influencing a given numerical target variable.
    Uses a Random Forest Regressor to compute feature importance's."""
    print("- Checking for feature importance.")
    # Ensure the target variable is in the numerical columns before proceeding
    if target_variable in numerical_columns:
        # Define feature matrix X (excluding the target variable) and target variable y
        X = df[numerical_columns].drop(columns=[target_variable])
        y = df[target_variable]
        # Train a Random Forest model to assess feature importance
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        # Extract feature importances
        importances = model.feature_importances_
        # Sort features by importance in descending order and retain the top_n features
        feature_importances = sorted(
            zip(X.columns, importances), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        # Create a dictionary to store details of top feature importances
        importance_details = {
            feature: {
                'importance_value': importance,
                'relative_rank': rank + 1
            }
            for rank, (feature, importance) in enumerate(feature_importances)
        }
        # Append the identified feature importance relation to the relations list
        relations.append({
            'attributes': [f[0] for f in feature_importances],
            'relation_type': 'feature_importance',
            'details': {
                'importances': importance_details,
                'target_variable': target_variable
            }
        })

def outlier_relationships(df, numerical_columns, relations, z_score_threshold=3.0, min_outlier_ratio=0.01, max_outlier_ratio=0.05, correlation_diff_threshold=0.3):
    """Identifies relationships between numerical features based on outlier patterns.
    It detects outliers using the Z-score method and examines how correlations change between
    normal data and outlier data."""
    print("- Checking for outliers relation.")
    for col in numerical_columns:
        # compute Z-scores for the column
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        # identify outliers based on the Z-score threshold
        outliers = df[z_scores > z_score_threshold]
        # compute the ratio of outliers in the dataset
        outlier_ratio = len(outliers) / len(df)

        # Consider the column only if the outlier ratio falls within the defined range        
        if min_outlier_ratio < outlier_ratio < max_outlier_ratio:
            # Compare the correlation of the outlier data with other numerical features
            for other_col in numerical_columns:
                if col != other_col:
                    outlier_correlation = outliers[col].corr(outliers[other_col])
                    normal_correlation = df[col].corr(df[other_col])
                    # Ensure correlations are not None before proceeding
                    if outlier_correlation is not None and normal_correlation is not None:
                        # Check if the difference in correlation is significant
                        if abs(outlier_correlation - normal_correlation) > correlation_diff_threshold:
                            relations.append({
                                'attributes': [col, other_col],
                                'relation_type': 'outlier_pattern',
                                'details': {
                                    'outlier_correlation': outlier_correlation,
                                    'normal_correlation': normal_correlation,
                                    'outlier_count': len(outliers)
                                }
                            })

def target_variable_analysis(df, target_variable, relations, z_score_threshold=3.0):
    """    Analyzes the target variable for outliers and its best-fitting distribution."""
    print("- Checking for target variable.")
    target_data = df[target_variable]
    # compute Z-scores for the target variable to detect outliers
    z_scores = np.abs((target_data - target_data.mean()) / target_data.std())
    # identify outliers based on the Z-score threshold
    outliers = target_data[z_scores > z_score_threshold]
    
    outlier_ratio = len(outliers) / len(target_data)

    # list of probability distributions to test for best fit
    distribution_types = ['norm', 'lognorm', 'expon', 'gamma', 'beta']
    best_fit = None
    best_p_value = 0

    # Iterate over different probability distributions to find the best fit
    for dist_name in distribution_types:
        # get the distribution function from scipy.stats
        dist = getattr(stats, dist_name)
        # fit the distribution to the target variable data
        params = dist.fit(target_data)
        #pPerform the Kolmogorov-Smirnov (KS) test to check the goodness-of-fit
        ks_stat, p_value = stats.kstest(target_data, dist_name, args=params)
        # update the best-fitting distribution based on the highest p-value
        if p_value > best_p_value:
            best_fit = dist_name
            best_p_value = p_value
    # append the target variable analysis to the relations list
    relations.append({
        'attributes': [target_variable],
        'relation_type': 'target_analysis',
        'details': {
            'outlier_ratio': outlier_ratio,
            'outlier_count': len(outliers),
            'distribution_type': best_fit,
            'distribution_p_value': best_p_value
        }
    })

def find_relations(df, target_variable, dataset_types):
    # creating list of relations
    relations = []
    # separating the columns to different types
    numerical_columns = [col for col, col_type in dataset_types.items() if col_type in ['integer', 'float']]
    categorical_columns = [col for col, col_type in dataset_types.items() if col_type in ['categorical_int', 'categorical_string']]
    datetime_columns = [col for col, col_type in dataset_types.items() if col_type == 'datetime']
    categorical_int_columns = [col for col, col_type in dataset_types.items() if col_type == 'categorical_int']

    # Get the relations with high correlation
    correlation_relations(df, numerical_columns, target_variable, relations)

    # Get the relations with the target value
    correlation_target_value(df, numerical_columns, target_variable, relations)

    # Get the relations with categorical features
    categorical_effects(df, categorical_columns, numerical_columns, target_variable, relations)

    # Get categorical relations using chi-square test
    chi_squared_relationship(df, categorical_columns, relations)

    # Get relation between date attribute and numerical attributes
    date_numerical_relationship(df, datetime_columns, numerical_columns, relations)

    # Get relations between date attribute and categorical attributes
    date_categorical_relationship(df, datetime_columns, categorical_columns, relations)

    # Get non-linear relations between attributes
    non_linear_relationships(df, numerical_columns, relations)

    # Get attributes importance using random forest
    feature_importance_relations(df, numerical_columns + categorical_int_columns, target_variable, relations)

    # Get outliers relations
    outlier_relationships(df, numerical_columns, relations)
    
    # Get the distribution of the target variable
    target_variable_analysis(df, target_variable, relations)

    return relations

def main():
    # Main function if you want to run only the relation detection algorithm.
    dataset_path = "Final Project/Datasets_Testing/AB_NYC_2019.csv"
    # input("Please enter the path to your Dataset: ")
    index_col = "id"
    # input("Please enter the index column: ")
    target_value = "price"
    # input("Please enter the name of your target value: ")
    df = read_data(dataset_path, index_col)
    if df is None:
        return
    # Understanding the types of columns in the data in order to create better visualizations.
    dataset_types = get_column_types(df)
    # Calling method to get the relations in the data
    find_relations(df, target_value, dataset_types)
    


if __name__ == "__main__":
    main()