import pandas as pd
import numpy as np
import pickle
import os.path
from automated_tool import get_column_types

# ratings_pd = pd.DataFrame({'bar_chart':[0,0],'line_chart':[0,0],'scatter_plot':[0,0]}, index=['0','1'])


visualization_types = {
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
        "use_cases": ["correlation matrix", "density", "patterns"],
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
        "use_cases": ["financial analysis", "price movement", "stock market", "trading"],
        "data_types": ["numerical", "time series"],
        "dimensions": [1],
        "min_data_points": 5,
        "required_metrics": ["open", "high", "low", "close"],
        "typical_fields": ["date", "open", "high", "low", "close", "volume"]
    },
    "multiple_boxplots": {
        "description": "Displays multiple boxplots for comparing distributions across different categories or features.",
        "use_cases": ["distribution comparison", "outlier detection", "feature comparison"],
        "data_types": ["numerical", "categorical"],
        "dimensions": ['all'],
        "min_data_points": 10,
        "min_features": 2
    },
    "multiple_violin_plots": {
        "description": "Shows density distributions for multiple features or categories with violin shapes.",
        "use_cases": ["distribution comparison", "density visualization", "feature comparison"],
        "data_types": ["numerical", "categorical"],
        "dimensions": ['all'],
        "min_data_points": 20,
        "min_features": 2
    },
    "covariance_heatmap": {
        "description": "Visualizes the covariance matrix between multiple numerical features.",
        "use_cases": ["correlation analysis", "feature relationships", "multivariate analysis"],
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
    # print('\n\n')
    # print(data_analysis)
    # print('\n\n')
    recommendations = []
    
    # Filter by analysis goal if provided
    candidate_viz_types = list(visualization_types.keys())
    if analysis_goal and analysis_goal in goal_to_viz:
        candidate_viz_types = goal_to_viz[analysis_goal]
    
    for viz_type in candidate_viz_types:
        viz_info = visualization_types[viz_type]
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
            score -= 5
            explanation.append(f"Data dimensions ({data_analysis['columns']}) don't visualization requirements")
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
            score += 3
            explanation.append(f"Data types ({', '.join(available_types)}) include required types ({', '.join(required_types)})")
        else:
            score -= 5
            explanation.append(f"Data types ({', '.join(available_types)}) don't include all required types ({', '.join(required_types)})")
        
        # Check number of data points
        if "min_data_points" in viz_info and data_analysis["rows"] < viz_info["min_data_points"]:
            score -= 2
            explanation.append(f"Dataset has fewer rows ({data_analysis['rows']}) than recommended ({viz_info['min_data_points']})")
            
        # Check categorical constraints
        if "max_categories" in viz_info:
            max_categories = viz_info["max_categories"]
            for col in data_analysis["categorical_columns"]:
                if data_analysis["unique_values"][col] > max_categories:
                    score -= 1
                    explanation.append(f"Column '{col}' has too many categories ({data_analysis['unique_values'][col]} > {max_categories})")
        
        
        
        # Add recommendation if score is positive
        if score > 0:
            recommendations.append({
                "type": viz_type,
                "score": score,
                "description": viz_info["description"],
                "explanation": explanation
            })
    
    # Sort recommendations by score
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    return recommendations

def generate_code_example(data, viz_type):
    """
    Generate sample code for the specified visualization type.
    """

    data_analysis = analyze_data(data)

    code = f"# Example code for creating a {viz_type}\n"
    code += "import matplotlib.pyplot as plt\n"
    code += "import seaborn as sns\n\n"
    
    if viz_type == "bar_chart":
        cat_col = data_analysis["categorical_columns"][0] if data_analysis["categorical_columns"] else data.columns[0]
        num_col = data_analysis["numerical_columns"][0] if data_analysis["numerical_columns"] else data.columns[1]
        code += f"plt.figure(figsize=(10, 6))\n"
        code += f"sns.barplot(x='{cat_col}', y='{num_col}', data=data)\n"
        code += "plt.title('Bar Chart')\n"
        code += "plt.xticks(rotation=45)\n"
        code += "plt.tight_layout()\n"
        code += "plt.show()"
        
    elif viz_type == "line_chart":
        x_col = data_analysis["datetime_columns"][0] if data_analysis["datetime_columns"] else data.columns[0]
        y_col = data_analysis["numerical_columns"][0] if data_analysis["numerical_columns"] else data.columns[1]
        code += f"plt.figure(figsize=(10, 6))\n"
        code += f"plt.plot(data['{x_col}'], data['{y_col}'])\n"
        code += "plt.title('Line Chart')\n"
        code += "plt.xlabel('{x_col}')\n"
        code += "plt.ylabel('{y_col}')\n"
        code += "plt.grid(True)\n"
        code += "plt.show()"
        
        
    return code


def save_ratings(ratings, file_name):
    with open(file_name+'.pkl', 'wb') as f:
        pickle.dump(ratings, f)

def load_ratings(file_name):
    file = file_name+'.pkl'
    if os.path.isfile(file):
        with open( file, 'rb') as f:
            ratings = pickle.load(f)
    else:
        ratings = pd.DataFrame({})
        for vis_type in visualization_types:
            if vis_type not in ratings.columns:
                ratings[vis_type] = np.nan

    
    return ratings




if __name__ == "__main__":
    # Get the currec=nt user ratings
    rating_string = "Please rate this visualization between 1 (Least helpfull) and 5 (Most helpfull).\n"

    ratings = load_ratings('user_ratings')
    print(f'\n{ratings}\n')
    user_id = input("Please enter a user id:\n")

    if not user_id in ratings.index:
        ratings.loc[user_id] = np.nan
        save_ratings(ratings, 'user_ratings')


    # Sample dataset
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'value': [10, 15, 7, 12, 18, 9, 14, 20, 11, 16, 22, 13],
        'count': [100, 150, 70, 120, 180, 90, 140, 200, 110, 160, 220, 130]
    })

    # Get recommendations
    recommendations = recommend(data[['category','value']])

    # Print recommendations
    print("Recommended visualizations:")
    for i, rec in enumerate(recommendations[:3]):
        user_rating = 0
        if pd.notna(ratings.loc[user_id, rec['type']]):
            user_rating = ratings.loc[user_id, rec['type']]

        print(f"\n{i+1}. {rec['type'].replace('_', ' ').title()}")
        print(f"   Description: {rec['description']}")
        print(f"   Score: {rec['score']/2 + user_rating}")
        print(f"   Rationale:")
        for exp in rec['explanation']:
            print(f"   - {exp}")
        
        new_rating = int(input(rating_string))
        if user_rating:
            ratings.loc[user_id, rec['type']] = user_rating * 0.8 + new_rating* 0.2
        else:
            ratings.loc[user_id, rec['type']] = new_rating
            
        save_ratings(ratings, 'user_ratings')    



    print(f'\n {ratings} \n')

