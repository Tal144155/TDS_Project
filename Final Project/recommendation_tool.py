import pandas as pd
import numpy as np
import pickle
import os.path
from sklearn.metrics.pairwise import pairwise_distances
from relation_detection_algorithm import get_column_types
from relation_detection_algorithm import find_relations

RATING_STRING = "Please rate this visualization between 1 (Least helpful) and 5 (Most helpful).\n"

RELATION_TYPES = {
    "high_correlation": {
        "description": "Identifies pairs of numerical features that have a strong linear relationship, indicating potential multicollinearity or redundancy in the dataset.",
        "use_cases": [
            "Feature selection",
            "Dimensionality reduction",
            "Understanding feature interactions"
        ],
        "data_types": ["numerical"],
        "dimensions": [2],
    },
    'target_correlation': {
        "description": "Measures the linear relationship between individual features and the target variable, helping to identify the most influential predictors.",
        "use_cases": [
            "Feature importance ranking",
            "Predictive modeling",
            "Feature selection"
        ],
        "data_types": ["numerical"],
        "dimensions": [2],
    },
    'categorical_effect': {
        "description": "Evaluates the statistical significance of categorical variables' impact on a numerical target variable using one-way ANOVA test.",
        "use_cases": [
            "Feature significance testing",
            "Group comparison",
            "Categorical feature importance"
        ],
        "data_types": ["categorical", "numerical"],
        "dimensions": [2],
    },
    'chi_squared': {
        "description": "Identifies statistically significant relationships between categorical variables using the chi-squared independence test.",
        "use_cases": [
            "Feature dependency analysis",
            "Categorical variable interaction detection",
            "Feature selection"
        ],
        "data_types": ["categorical"],
        "dimensions": [2],
    },
    'date_numerical_trend': {
        "description": "Detects temporal trends in numerical features by measuring their correlation with time-based attributes.",
        "use_cases": [
            "Time series analysis",
            "Trend identification",
            "Temporal pattern recognition"
        ],
        "data_types": ["numerical", "time series"],
        "dimensions": [2],
    },
    'date_categorical_distribution': {
        "description": "Analyzes how categorical variable distributions change or are distributed across different time periods.",
        "use_cases": [
            "Temporal categorical pattern detection",
            "Seasonal variation analysis",
            "Time-based segmentation"
        ],
        "data_types": ["categorical", "time series"],
        "dimensions": [2],
    },
    'non_linear': {
        "description": "Identifies complex, non-linear relationships between numerical features using mutual information score.",
        "use_cases": [
            "Advanced feature interaction detection",
            "Non-linear dependency analysis",
            "Complex relationship mapping"
        ],
        "data_types": ["numerical"],
        "dimensions": [2],
    },
    'feature_importance': {
        "description": "Ranks features based on their predictive power using a Random Forest Regressor's feature importance metric.",
        "use_cases": [
            "Predictive modeling",
            "Feature selection",
            "Model interpretability"
        ],
        "data_types": ["numerical"],
        "dimensions": [2],
    },
    'outlier_pattern': {
        "description": "Detects unique correlation patterns among outliers that differ from the overall dataset's correlations.",
        "use_cases": [
            "Anomaly detection",
            "Robust correlation analysis",
            "Outlier impact assessment"
        ],
        "data_types": ["numerical"],
        "dimensions": [2],
    },
    'cluster_group': {
        "description": "Identifies groups of features that exhibit similar clustering characteristics based on their importance within specific clusters.",
        "use_cases": [
            "Feature grouping",
            "Dimensionality reduction",
            "Structural data understanding"
        ],
        "data_types": ["numerical"],
        "dimensions": [1],
    },
    'target_analysis': {
        "description": "Provides a comprehensive analysis of the target variable, including outlier characteristics and distribution properties.",
        "use_cases": [
            "Target variable understanding",
            "Distribution fitting",
            "Outlier detection"
        ],
        "data_types": ["numerical"],
        "dimensions": [1],
    }
}



# Save the user ratings to a pickle file for keeping the progress and assessing our model
def save_ratings(ratings, file_name):
    with open(file_name+'.pkl', 'wb') as f:
        pickle.dump(ratings, f)

# Load user ratings for collaborative filtering 
def load_ratings(file_name, rec_types):
    file = file_name+'.pkl'
    if os.path.isfile(file):
        with open( file, 'rb') as f:
            ratings = pickle.load(f)
    else:
        ratings = pd.DataFrame({})
        for type in rec_types:
            if type not in ratings.columns:
                ratings[type] = np.nan

    return ratings


def CFUB(ratings_pd):
    # Get the mean rating for each user
    ratings = ratings_pd.to_numpy()
    mean_user_rating = ratings_pd.mean(axis=1).to_numpy().reshape(-1, 1)
    # calculate the similarity between users
    ratings_diff = (ratings - mean_user_rating)
    ratings_diff[np.isnan(ratings_diff)]=4
    user_similarity = 1-pairwise_distances(ratings_diff, metric='cosine')
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return pred

def CFCB(ratings_pd):
    # Get the mean rating for each user
    ratings = ratings_pd.to_numpy()
    mean_user_rating = ratings_pd.mean(axis=1).to_numpy().reshape(-1, 1)
    # calculate the similarity between visualizations
    ratings_diff = (ratings - mean_user_rating)
    ratings_diff[np.isnan(ratings_diff)]=4
    vis_similarity = 1-pairwise_distances(ratings_diff, metric='cosine')
    pred = mean_user_rating + vis_similarity.dot(ratings_diff) / np.array([np.abs(vis_similarity).sum(axis=1)]).T
    return pred

# Weighted sum of two predictions
def combine_pred(pred1, pred2, w1 = 0.5, w2 = 0.5):
    # Replace NaN values with 0
    pred1 = np.nan_to_num(pred1, nan=4.0)
    pred2 = np.nan_to_num(pred2, nan=4.0)

    return w1 * pred1 + w2 * pred2


def normalize_score(value, metric_type):
    """
    Normalize different types of statistical measures to a 1-5 scale.
    """
    # Normalization strategies for different metric types
    normalization_strategies = {
        'high_correlation': {
            'abs_range': (0.5, 1.0),  # Correlation values are between -1 and 1
            'percentile_thresholds': [0.5, 0.7, 0.8, 0.9]
        },
        'target_correlation': {
            'abs_range': (0.5, 1.0),  # Correlation with target variable
            'percentile_thresholds': [0.5, 0.6, 0.7, 0.9]
        },
        'categorical_effect': {
            'abs_range': (0, 0.05),  # P-values, lower is stronger
            'percentile_thresholds': [0.05, 0.02, 0.01, 0.009]
        },
        'chi_squared': {
            'abs_range': (0, 0.05),  # P-values, lower is stronger
            'percentile_thresholds': [0.05, 0.02, 0.01, 0.009]
        },
        'date_numerical_trend': {
            'abs_range': (0.5, 1.0),  # Correlation values
            'percentile_thresholds': [0.5, 0.7, 0.8, 0.9]
        },
        'date_categorical_distribution':{
            'abs_range': (0, 0.05),  # P-values, lower is stronger
            'percentile_thresholds': [0.05, 0.02, 0.01, 0.009]
        },
        'non_linear': {
            'abs_range': (0.5, 1.0),  # Mutual information score
            'percentile_thresholds': [0.5, 0.7, 0.8, 0.9]
        },
        'feature_importance': {
            'abs_range': (0, 1.0),  # Feature importance values
            'percentile_thresholds': [0.2, 0.4, 0.6, 0.8]
        },
        'outlier_pattern': {
            'abs_range': (0.3, 1.0),  # Correlation differences
            'percentile_thresholds': [0.3, 0.5, 0.7, 0.9]
        }
    }
    

    if metric_type == 'cluster_group':
        # Normalize based on number of features in cluster or importance
        return min(max(1, int(value * 5)), 5)
    elif metric_type == 'target_analysis':
        # Normalize outlier ratio or distribution significance
        return min(max(1, int(value * 5)), 5)
    
    # We'll set the middle score as the default
    if metric_type not in normalization_strategies:
        return 3  
    
    strategy = normalization_strategies[metric_type]
    
    # Absolute value for signed metrics
    abs_value = abs(value)
    
    
    # Value-based normalization
    min_val, max_val = strategy['abs_range']
    
    # Normalize to 1-5 range
    if abs_value <= min_val:
        return 1
    elif abs_value >= max_val:
        return 5
    else:
        normalized = 1 + 4 * (abs_value - min_val) / (max_val - min_val)
        return int(min(max(normalized, 1), 5))

# Example usage
def get_relation_scores(relations):
    """
    Apply strength normalization to all relations.
    """
    for relation in relations:
        # Choose the appropriate value for normalization based on relation type
        if relation['relation_type'] in {'high_correlation', 'target_correlation','date_numerical_trend',}:
            value = relation['details']['correlation_value']
        elif relation['relation_type'] in {'categorical_effect', 'date_categorical_distribution', 'chi_squared',  }:
            value = relation['details']['p_value']
        elif relation['relation_type'] == 'non_linear':
            value = relation['details']['mutual_information']
        # Add more specific conditions as needed
        else:
            value = 2 # Default fallback
        
        # 
        relation['score'] = normalize_score(
            value, 
            relation['relation_type']
        )
    
    return relations

def get_top_relations(relations):
    """
    Generate a pandas DF for the top relations of each type
    """
    algo_rec_df = pd.DataFrame({})
    top_relations = {}

    for i, rel in enumerate(relations):
        type = rel['relation_type']
        if not type in top_relations:
            top_relations[type] = {'score': rel['score'], 'index': i}

    for type in RELATION_TYPES:
        score = 0
        indx = -1
        if type in top_relations:
            score = top_relations[type]['score']
            indx = top_relations[type]['index']
        algo_rec_df.loc[0, type] = score
        algo_rec_df.loc[1, type] = indx

    return algo_rec_df