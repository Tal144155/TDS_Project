import pandas as pd
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from relation_detection_algorithm import get_column_types
from relation_detection_algorithm import find_relations

# ratings_pd = pd.DataFrame({'bar_chart':[0,0],'line_chart':[0,0],'scatter_plot':[0,0]}, index=['0','1'])
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

VISUALIZATION_TYPES = {
    "bar_chart": {
        "description": "Displays categorical data with rectangular bars.",
        "use_cases": ["comparing categories", "showing distribution", "ranking"],
        "data_types": ["categorical", "numerical"],
        "dimensions": [1, 2],
        "max_categories": 20
    },
    "line_chart": {
        "description": "Shows trends over a continuous variable, usually time.",
        "use_cases": ["trends", "time series", "continuous changes"],
        "data_types": ["numerical", "time series"],
        "dimensions": [1, 2],
        "min_data_points": 5
    },
    "scatter_plot": {
        "description": "Shows relationship between two numerical variables.",
        "use_cases": ["correlation", "distribution", "clusters"],
        "data_types": ["numerical"],
        "dimensions": [2, 3],
        "min_data_points": 10
    },
    "pie_chart": {
        "description": "Shows proportion of each category in a whole.",
        "use_cases": ["composition", "proportion"],
        "data_types": ["categorical"],
        "dimensions": [1],
        "max_categories": 7
    },
    "histogram": {
        "description": "Shows distribution of a numerical variable.",
        "use_cases": ["distribution", "frequency"],
        "data_types": ["numerical"],
        "dimensions": [1],
        "min_data_points": 20
    },
    "heatmap": {
        "description": "Shows magnitude of values across two dimensions using color.",
        "use_cases": ["correlation", "density", "patterns"],
        "data_types": ["numerical"],
        "dimensions": [2],
        "min_data_points": 9
    },
    "box_plot": {
        "description": "Shows distribution statistics with quartiles and outliers.",
        "use_cases": ["distribution", "outliers", "comparison"],
        "data_types": ["numerical", "categorical"],
        "dimensions": [1, 2],
        "min_data_points": 5
    },
    "treemap": {
        "description": "Shows hierarchical data as nested rectangles.",
        "use_cases": ["hierarchy", "proportion", "composition"],
        "data_types": ["hierarchical", "numerical"],
        "dimensions": [2, 3],
        "min_categories": 4
    },
    "candlestick_chart": {
        "description": "Shows price movements for financial instruments over time.",
        "use_cases": ["financial", "price", "stock", "trading"],
        "data_types": ["numerical", "time series"],
        "dimensions": [1],
        "min_data_points": 5,
        "required_metrics": ["open", "high", "low", "close"],
        "typical_fields": ["date", "open", "high", "low", "close", "volume"]
    },
    "multiple_boxplots": {
        "description": "Displays multiple boxplots for comparing distributions across different categories or features.",
        "use_cases": ["distribution", "outlier detection", "feature comparison"],
        "data_types": ["numerical", "categorical"],
        "dimensions": ['all'],
        "min_data_points": 10,
        "min_features": 2
    },
    "multiple_violin_plots": {
        "description": "Shows density distributions for multiple features or categories with violin shapes.",
        "use_cases": ["distribution", "density", "feature comparison"],
        "data_types": ["numerical", "categorical"],
        "dimensions": ['all'],
        "min_data_points": 20,
        "min_features": 2
    },
    "covariance_heatmap": {
        "description": "Visualizes the covariance matrix between multiple numerical features.",
        "use_cases": ["correlation", "feature relationships", "multivariate"],
        "data_types": ["numerical"],
        "dimensions": ['all'],
        "min_data_points": 10,
        "min_features": 2
    }
}

goal_to_viz = {
    "comparison": ["bar_chart", "scatter_plot", "box_plot", "radar_chart"],
    "distribution": ["histogram", "box_plot", "violin_plot", "density_plot"],
    "composition": ["pie_chart", "stacked_bar_chart", "treemap"],
    "relationship": ["scatter_plot", "bubble_chart", "heatmap", "line_chart"],
    "trends": ["line_chart",  "candlestick_chart"],
    "part_to_whole": ["pie_chart", "treemap", "stacked_bar_chart"],
    "ranking": ["bar_chart", "lollipop_chart", "bullet_chart"],
    "correlation": ["scatter_plot", "heatmap", "bubble_chart"],
    "flow": ["sankey_diagram", "network_diagram", "chord_diagram"]
}
    
def analyze_data(data):
    """
    Analyze the provided dataframe to determine its characteristics.
    """
    data_analysis = {
        "columns": data.shape[1],
        "rows": len(data),
        "column_types": {},
        "categorical_columns": [],
        "numerical_columns": [],
        "datetime_columns": [],
        "unique_values": {}
    }
    
    for col in data.columns:
        # Check data type
        if pd.api.types.is_numeric_dtype(data[col]):
            data_analysis["column_types"][col] = "numerical"
            data_analysis["numerical_columns"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            data_analysis["column_types"][col] = "datetime"
            data_analysis["datetime_columns"].append(col)
        else:
            data_analysis["column_types"][col] = "categorical"
            data_analysis["categorical_columns"].append(col)
        
        # Count unique values for each column
        data_analysis["unique_values"][col] = data[col].nunique()
        
        
            
    return data_analysis

def recommend(data, analysis_goal=None):
    """
    Recommend visualizations based on data and analysis goal.
    """

    data_analysis = analyze_data(data)
    recommendations = []
    
    # Filter by analysis goal if provided
    candidate_viz_types = list(VISUALIZATION_TYPES.keys())
    # if analysis_goal and analysis_goal in goal_to_viz:
    #     candidate_viz_types = goal_to_viz[analysis_goal]
    
    for viz_type in candidate_viz_types:
        viz_info = VISUALIZATION_TYPES[viz_type]
        score = 0
        explanation = []
        # Check data dimensions
        if data_analysis["columns"] in viz_info["dimensions"]:
            score += 3
            explanation.append(f"Data dimensions ({data_analysis['columns']}) match visualization requirements")
        elif 'all' in viz_info["dimensions"] and data_analysis["columns"] >= viz_info["min_features"]:
            score += 2
            explanation.append(f"Data dimensions ({data_analysis['columns']}) match visualization requirements")
        else:
            score = 0
            explanation.append(f"Data dimensions ({data_analysis['columns']}) don't visualization requirements")
            # Add recommendation to a list
            recommendations.append({
                "type": viz_type,
                "score": score,
                "description": viz_info["description"],
                "explanation": explanation
            })
            continue
        # Check data types
        required_types = set(viz_info["data_types"])
        available_types = set()
        if data_analysis["numerical_columns"]:
            available_types.add("numerical")
        if data_analysis["categorical_columns"]:
            available_types.add("categorical")
        if data_analysis["datetime_columns"]:
            available_types.add("time series")

            
        if required_types.issubset(available_types) or available_types.issubset(required_types) :
            score += 2
            explanation.append(f"Data types ({', '.join(available_types)}) include required types ({', '.join(required_types)})")
        else:
            score = 0 
            explanation.append(f"Data types ({', '.join(available_types)}) don't include all required types ({', '.join(required_types)})")
            # Add recommendation to a list
            recommendations.append({
                "type": viz_type,
                "score": score,
                "description": viz_info["description"],
                "explanation": explanation
            })
            continue
        
        # Check number of data points
        if "min_data_points" in viz_info and data_analysis["rows"] < viz_info["min_data_points"]:
            if  score >= 2:
                score -= 2
            else:
                score = 0
            explanation.append(f"Dataset has fewer rows ({data_analysis['rows']}) than recommended ({viz_info['min_data_points']})")
            
        # Check categorical constraints
        if "max_categories" in viz_info:
            max_categories = viz_info["max_categories"]
            for col in data_analysis["categorical_columns"]:
                if data_analysis["unique_values"][col] > max_categories:
                    if  score >= 1:
                        score -= 1
                    else:
                        score = 0
                    explanation.append(f"Column '{col}' has too many categories ({data_analysis['unique_values'][col]} > {max_categories})")
        
        
        
        # Add recommendation to a list
        recommendations.append({
            "type": viz_type,
            "score": score,
            "description": viz_info["description"],
            "explanation": explanation
        })

    
    return recommendations


# Save the user ratings to a pickle file for keeping the progress and assessing our model
def save_ratings(ratings, file_name):
    with open(file_name+'.pkl', 'wb') as f:
        pickle.dump(ratings, f)

# Load user ratings for content filtering 
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
    ratings_diff[np.isnan(ratings_diff)]=0
    user_similarity = 1-pairwise_distances(ratings_diff, metric='cosine')
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    pred.round(2)
    return pred

def CFCB(ratings_pd):
    # Get the mean rating for each user
    ratings = ratings_pd.to_numpy()
    mean_user_rating = ratings_pd.mean(axis=1).to_numpy().reshape(-1, 1)
    # calculate the similarity between visualizations
    ratings_diff = (ratings - mean_user_rating)
    ratings_diff[np.isnan(ratings_diff)]=0
    vis_similarity = 1-pairwise_distances(ratings_diff, metric='cosine')
    pred = mean_user_rating + vis_similarity.dot(ratings_diff) / np.array([np.abs(vis_similarity).sum(axis=1)]).T
    pred.round(2)
    return pred

# Weighted sum of two predictions
def combine_pred(pred1, pred2, w1 = 0.5, w2 = 0.5):
    # Replace NaN values with 0
    pred1 = np.nan_to_num(pred1, nan=0.0)
    pred2 = np.nan_to_num(pred2, nan=0.0)

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
            'abs_range': (0, 0.01),  # P-values, lower is stronger
            'percentile_thresholds': [0.01, 0.007, 0.005, 0.001]
        },
        'chi_squared': {
            'abs_range': (0, 0.01),  # P-values, lower is stronger
            'percentile_thresholds': [0.01, 0.007, 0.005, 0.001]
        },
        'date_numerical_trend': {
            'abs_range': (0.5, 1.0),  # Correlation values
            'percentile_thresholds': [0.5, 0.7, 0.8, 0.9]
        },
        'date_categorical_distribution':{
            'abs_range': (0, 0.01),  # P-values, lower is stronger
            'percentile_thresholds': [0.01, 0.007, 0.005, 0.001]
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
            value = 3.5  # Default fallback
        
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

if __name__ == "__main__":

    # Sample dataset
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'value': [8, 15, 7, 12, 18, 9, 14, 20, 11, 16, 22, 13],
        'count': [100, 150, 70, 120, 180, 90, 140, 200, 110, 160, 220, 130]
    })
    target_value = 'value'
    # Load user ratings from the pickle file
    ratings = load_ratings('user_ratings_rel', RELATION_TYPES)
    print(f'\n{ratings}\n')
    user_id = input("Please enter a user id:\n")

    # If this is a new user, add the user to the dataframe.
    if not user_id in ratings.index:
        ratings.loc[user_id] = np.nan
        save_ratings(ratings, 'user_ratings_rel')


     # Get automatic recommendations 
    dataset_types = get_column_types(data)
    algo_rec = find_relations(data, target_value, dataset_types)
    algo_rec = get_relation_scores(algo_rec)
    while True:
        if not algo_rec:
            print("Those are all the meaningful relations we've found.\n We hope you found this helpful! (:)")
            break
        # Get the current user ratings
        # ratings = load_ratings('user_ratings_rel', RELATION_TYPES)
        combined_user_vis_pred = combine_pred(CFCB(ratings), CFUB(ratings), 0.5, 0.5)

        # Make a df for the recommendation system
        algo_rec_df = get_top_relations(algo_rec)


        user_index = ratings.index.get_loc(user_id)
        recommendations = combine_pred(combined_user_vis_pred[user_index], algo_rec_df.to_numpy()[0], 0.7, 0.3)
        
        # Print the top recommendations sorted by score
        print("Recommended visualizations:")
        index = int(algo_rec_df.iloc[1,recommendations.argmax()])
        rec = algo_rec.pop(index)
        user_rating = 0
        if pd.notna(ratings.loc[user_id, rec['relation_type']]):
            user_rating = ratings.loc[user_id, rec['relation_type']]

        print(f"\n    {rec['relation_type'].replace('_', ' ').title()}")
        print(f"   Description: {RELATION_TYPES[rec['relation_type']]['description']}")
        print(f"   Score: {recommendations[index]}")
        print(f"   Rationale:")
        for exp in rec['details']:
            print(f"   - {exp}")
        
        new_rating = int(input(RATING_STRING))
        if user_rating:
            ratings.loc[user_id, rec['relation_type']] = user_rating * 0.8 + new_rating* 0.2
        else:
            ratings.loc[user_id, rec['relation_type']] = new_rating

        save_ratings(ratings, 'user_ratings_rel')    



        print(f'\n {ratings} \n')

