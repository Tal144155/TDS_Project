import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import chi2_contingency, f_oneway
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats


import warnings
warnings.filterwarnings('ignore')

TOP_N_RELATIONS = 10

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
    print("- Checking for correlation.")
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
    print("- Checking for correlation with the target variable.")
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

def categorical_effects(df, categorical_columns, numerical_columns, target_variable, relations, p_value_threshold=0.01):
    print("- Checking for categorical effect.")
    temp_relations = []
    if target_variable in numerical_columns:
        for cat_feature in categorical_columns:
            groups = [df[df[cat_feature] == cat][target_variable].dropna() for cat in df[cat_feature].unique()]
            if len(groups) > 1:
                f_stat, p_value = f_oneway(*groups)
                if p_value < p_value_threshold:
                    temp_relations.append(
                        {'attributes': [cat_feature, target_variable],
                         'relation_type': 'categorical_effect',
                         'details': {'p_value': p_value}})
    temp_relations.sort(key=lambda x: x['details']['p_value'])
    relations.extend(temp_relations[:TOP_N_RELATIONS])


def chi_squared_relationship(df, categorical_columns, relations, p_value_threshold=0.01):
    print("- Checking for chi square relation.")
    temp_relations = []
    for i, feature1 in enumerate(categorical_columns):
        for feature2 in categorical_columns[i + 1:]:
            contingency_table = pd.crosstab(df[feature1], df[feature2])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            if p < p_value_threshold:
                temp_relations.append(
                    {'attributes': [feature1, feature2],
                     'relation_type': 'chi_squared',
                     'details': {'p_value': p}})
    temp_relations.sort(key=lambda x: x['details']['p_value'])
    relations.extend(temp_relations[:TOP_N_RELATIONS])


# Function to check for numerical feature trends over time
def date_numerical_relationship(df, date_columns, numerical_columns, relations, correlation_threshold=0.5):
    print("- Checking for date with numerical variables.")
    temp_relations = []
    for date_col in date_columns:
        # Safely convert to datetime and drop NaT values
        valid_dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if valid_dates.empty:
            continue
        df['time_ordinal'] = valid_dates.map(pd.Timestamp.toordinal)
        for num_feature in numerical_columns:
            # Only use rows where the date is valid
            valid_data = df.loc[valid_dates.index, num_feature].dropna()
            if not valid_data.empty:
                corr_value = df.loc[valid_data.index, 'time_ordinal'].corr(valid_data)
                if abs(corr_value) > correlation_threshold:
                    temp_relations.append(
                        {'attributes': [date_col, num_feature],
                         'relation_type': 'date_numerical_trend',
                         'details': {'correlation_value': corr_value}}
                    )
    temp_relations.sort(key=lambda x: abs(x['details']['correlation_value']), reverse=True)
    relations.extend(temp_relations[:TOP_N_RELATIONS])


# Function to check for categorical feature distribution over date features
def date_categorical_relationship(df, date_columns, categorical_columns, relations, p_value_threshold=0.01):
    print("- Checking for date with categorical variable.")
    temp_relations = []
    for date_col in date_columns:
        df['date_period'] = pd.to_datetime(df[date_col]).dt.to_period('M')
        for cat_feature in categorical_columns:
            contingency_table = pd.crosstab(df['date_period'], df[cat_feature])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            if p < p_value_threshold:
                temp_relations.append(
                    {'attributes': [date_col, cat_feature],
                     'relation_type': 'date_categorical_distribution',
                     'details': {'p_value': p}}
                )
    temp_relations.sort(key=lambda x: x['details']['p_value'])
    relations.extend(temp_relations[:TOP_N_RELATIONS])

def non_linear_relationships(df, numerical_columns, relations, threshold=0.5):    
    print("- Checking for non linear relation.")
    for col1 in numerical_columns:
        for col2 in numerical_columns:
            if col1 != col2:
                mi = mutual_info_score(
                    pd.qcut(df[col1], 10, duplicates='drop', labels=False), 
                    pd.qcut(df[col2], 10, duplicates='drop', labels=False)
                )
                if mi > threshold:
                    relations.append({
                        'attributes': [col1, col2],
                        'relation_type': 'non_linear',
                        'details': {'mutual_information': mi}
                    })

def feature_importance_relations(df, numerical_columns, target_variable, relations, top_n=5):
    print("- Checking for feature importance.")
    if target_variable in numerical_columns:
        X = df[numerical_columns].drop(columns=[target_variable])
        y = df[target_variable]
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        
        feature_importances = sorted(
            zip(X.columns, importances), 
            key=lambda x: x[1], 
            reverse=True
        )
        for feature, importance in feature_importances[:top_n]:
            relations.append({
                'attributes': [feature, target_variable],
                'relation_type': 'feature_importance',
                'details': {
                    'importance_value': importance,
                    'relative_rank': feature_importances.index((feature, importance)) + 1
                }
            })

def outlier_relationships(df, numerical_columns, relations, z_score_threshold=3.0, min_outlier_ratio=0.01, max_outlier_ratio=0.05, correlation_diff_threshold=0.3):
    print("- Checking for outliers relation.")
    for col in numerical_columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > z_score_threshold]
        
        outlier_ratio = len(outliers) / len(df)
        
        if min_outlier_ratio < outlier_ratio < max_outlier_ratio:
            for other_col in numerical_columns:
                if col != other_col:
                    outlier_correlation = outliers[col].corr(outliers[other_col])
                    normal_correlation = df[col].corr(df[other_col])
                    
                    if outlier_correlation is not None and normal_correlation is not None:
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


def cluster_feature_relations(df, numerical_columns, relations, max_clusters=10, feature_importance_threshold=0.1):
    print("- Checking for cluster relation.")
    if len(numerical_columns) < 3:
        return
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_columns])
    
    best_n_clusters = 2
    best_score = -1
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    
    cluster_centers = kmeans.cluster_centers_
    
    for cluster_idx in range(best_n_clusters):
        center = cluster_centers[cluster_idx]
        feature_importance = np.abs(center)
        
        selected_indices = np.where(feature_importance > feature_importance_threshold)[0]
        selected_features = [numerical_columns[i] for i in selected_indices]
        
        if len(selected_features) >= 2:
            relations.append({
                'attributes': selected_features,
                'relation_type': 'cluster_group',
                'details': {
                    'cluster_id': cluster_idx,
                    'importance_scores': {
                        numerical_columns[i]: feature_importance[i] for i in selected_indices
                    }
                }
            })


def target_variable_analysis(df, target_variable, relations, z_score_threshold=3.0):
    print("- Checking for target variable.")
    target_data = df[target_variable]
    z_scores = np.abs((target_data - target_data.mean()) / target_data.std())
    outliers = target_data[z_scores > z_score_threshold]
    
    outlier_ratio = len(outliers) / len(target_data)
    
    distribution_types = ['norm', 'lognorm', 'expon', 'gamma', 'beta']
    best_fit = None
    best_p_value = 0
    
    for dist_name in distribution_types:
        dist = getattr(stats, dist_name)
        params = dist.fit(target_data)
        ks_stat, p_value = stats.kstest(target_data, dist_name, args=params)
        
        if p_value > best_p_value:
            best_fit = dist_name
            best_p_value = p_value
    
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
    relations = []
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

    # Get clusters relations
    cluster_feature_relations(df, numerical_columns, relations)
    
    # Get the distribution of the target variable
    target_variable_analysis(df, target_variable, relations)

    return relations



def main():
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