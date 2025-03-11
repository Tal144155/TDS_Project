import pandas as pd
import numpy as np
import os.path
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plot_generator import *
from recommendation_tool import *
from relation_detection_algorithm import *
import warnings
warnings.filterwarnings('ignore')

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

def create_visualizations(dataset_path, rating_location, target_variable, index = None):
    df = read_data(dataset_path)
    ratings = load_ratings(rating_location, RELATION_TYPES)
    user_id = input("Please enter a user id:\n")

    # If this is a new user, add the user to the dataframe.
    if not user_id in ratings.index:
        ratings.loc[user_id] = np.nan
        save_ratings(ratings, rating_location)

    # Get automatic recommendations 
    dataset_types = get_column_types(df)
    algo_rec = find_relations(df, target_variable, dataset_types)

    algo_rec = get_relation_scores(algo_rec)
    plot_index = 0
    num_chose_outliers=0

    while (plot_index < 6):
        if algo_rec:
                # get the most wanted visualization by the user
                combined_user_vis_pred = combine_pred(CFIB(ratings), CFUB(ratings), 0.5, 0.5)
                algo_rec_df = get_top_relations(algo_rec)
                user_index = ratings.index.get_loc(user_id)
                recommendations = combine_pred(combined_user_vis_pred[user_index], algo_rec_df.to_numpy()[0], 0.7, 0.3)
                index = int(algo_rec_df.iloc[1,recommendations.argmax()])
                chosen_plot = algo_rec.pop(index)
                if chosen_plot['relation_type'] == "outlier_pattern" and num_chose_outliers >= 2:
                    # get other relations instead until we get the relation that is not outlier
                    while chosen_plot['relation_type'] == "outlier_pattern" and algo_rec:
                        combined_user_vis_pred = combine_pred(CFIB(ratings), CFUB(ratings), 0.5, 0.5)
                        algo_rec_df = get_top_relations(algo_rec)
                        user_index = ratings.index.get_loc(user_id)
                        recommendations = combine_pred(combined_user_vis_pred[user_index], algo_rec_df.to_numpy()[0], 0.7, 0.3)
                        index = int(algo_rec_df.iloc[1,recommendations.argmax()])
                        chosen_plot = algo_rec.pop(index)
                # otherwise present this visualization
                elif chosen_plot['relation_type'] == "outlier_pattern":
                    num_chose_outliers += 1
                plot_save_name = f'plot_{plot_index}'
                # Based on the relation, call the correct method from the Plot Generator
                if chosen_plot['relation_type'] == "high_correlation":
                    plot_high_correlation(df, chosen_plot["attributes"][0], chosen_plot["attributes"][1], chosen_plot["details"]["correlation_value"], plot_save_name)
                elif chosen_plot['relation_type'] == "target_correlation":
                    plot_target_correlation(df, chosen_plot["attributes"][0], chosen_plot["attributes"][1], chosen_plot["details"]["correlation_value"], plot_save_name)
                elif chosen_plot['relation_type'] == "categorical_effect":
                    plot_categorical_effect(df, chosen_plot["attributes"][0], chosen_plot["attributes"][1], chosen_plot["details"]["p_value"], plot_save_name)
                elif chosen_plot['relation_type'] == "chi_squared":
                    plot_chi_squared(df, chosen_plot["attributes"][0], chosen_plot["attributes"][1], chosen_plot["details"]["p_value"], plot_save_name)
                elif chosen_plot['relation_type'] == "date_numerical_trend":
                    plot_date_numerical_trend(df, chosen_plot["attributes"][0], chosen_plot["attributes"][1], chosen_plot["details"]["correlation_value"], plot_save_name)
                elif chosen_plot['relation_type'] == "date_categorical_distribution":
                    plot_date_categorical_distribution(df, chosen_plot["attributes"][0], chosen_plot["attributes"][1], chosen_plot["details"]["p_value"], plot_save_name)
                elif chosen_plot['relation_type'] == "non_linear":
                    plot_non_linear(df, chosen_plot["attributes"][0], chosen_plot["attributes"][1], chosen_plot["details"]["mutual_information"], plot_save_name)
                elif chosen_plot['relation_type'] == "feature_importance":
                    plot_feature_importance(df, chosen_plot["details"]["importances"], chosen_plot["details"]["target_variable"], plot_save_name)
                elif chosen_plot['relation_type'] == "outlier_pattern":
                    plot_outlier_pattern(df, chosen_plot["attributes"][0], chosen_plot["attributes"][1], plot_save_name)
                elif chosen_plot['relation_type'] == "target_analysis":
                    plot_target_analysis(df, chosen_plot["attributes"][0], chosen_plot["details"]["outlier_ratio"], chosen_plot["details"]["distribution_type"], plot_save_name)

                image_path = os.path.abspath(os.path.join(PLOTS_DIR, f'{plot_save_name}.png'))
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        new_rating = 0
        # receiving rating from the user and updating accordingly
        while new_rating > 5 or new_rating < 1:
            new_rating = int(input('please enter a rating between 1 (least helpfull) and 5 (most helpfull)')) 
        user_rating = 0
        if pd.notna(ratings.loc[user_id, chosen_plot['relation_type']]):
            user_rating = ratings.loc[user_id, chosen_plot['relation_type']]
        if user_rating:
            ratings.loc[user_id, chosen_plot['relation_type']] = user_rating * 0.5 + new_rating* 0.5
        else:
            ratings.loc[user_id, chosen_plot['relation_type']] = new_rating

        save_ratings(ratings, rating_location)   
        plot_index+=1 
        ratings = load_ratings(rating_location, RELATION_TYPES)

    print("process finished")


create_visualizations("Final Project/Datasets_Testing/movie_new.csv", "user_ratings", "revenue")