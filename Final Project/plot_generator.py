import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def save_plot(fig, plot_name):
    # Saving the plot in the destination directory, with the name chosen
    file_path = os.path.join(PLOTS_DIR, f"{plot_name}.png")
    fig.savefig(file_path)
    plt.close(fig)
    print(f"Saved plot: {file_path}")

def plot_high_correlation(df, feature1, feature2, correlation_value, plot_name):
    # Plotting for high correlation, adding trend line to show the connection between them
    fig, ax = plt.subplots()
    sns.regplot(x=df[feature1], y=df[feature2], scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'color': 'red'})
    ax.set_title(f'High Correlation: {feature1} vs {feature2} (Corr: {correlation_value:.2f})')
    save_plot(fig, plot_name)

def plot_target_correlation(df, feature, target, correlation_value, plot_name):
    # Plotting for high correlation with the target value
    fig, ax = plt.subplots()
    sns.regplot(x=df[feature], y=df[target], scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'color': 'red'})
    ax.set_title(f'Target Correlation: {feature} vs {target} (Corr: {correlation_value:.2f})')
    save_plot(fig, plot_name)

def plot_categorical_effect(df, categorical_feature, target, p_value, plot_name):
    # Plotting for categorical effect, if we have categorical feature with non categorical feature
    fig, ax = plt.subplots()
    sns.boxplot(x=df[categorical_feature], y=df[target], ax=ax)
    ax.set_title(f'Categorical Effect: {categorical_feature} on {target} (p-value: {p_value:.4f})')
    save_plot(fig, plot_name)

def plot_chi_squared(df, feature1, feature2, p_value, plot_name):
    # Plotting for connection between 2 categorical features using heatmap
    contingency_table = pd.crosstab(df[feature1], df[feature2])
    fig, ax = plt.subplots()
    sns.heatmap(contingency_table, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title(f'Chi-Squared Test: {feature1} vs {feature2} (p-value: {p_value:.4f})')
    save_plot(fig, plot_name)

def plot_date_numerical_trend(df, date_col, num_feature, correlation_value, plot_name):
    # Plotting for date numerical connection with line plot to show the connection
    fig, ax = plt.subplots()
    sns.lineplot(x=df[date_col], y=df[num_feature], ax=ax)
    ax.set_title(f'Date Numerical Trend: {date_col} vs {num_feature} (Corr: {correlation_value:.2f})')
    save_plot(fig, plot_name)

def plot_date_categorical_distribution(df, date_col, cat_feature, p_value, plot_name):
    # Plotting for date categorical connection using bar plot to show progress
    contingency_table = pd.crosstab(df[date_col].dt.to_period('M'), df[cat_feature])
    fig, ax = plt.subplots(figsize=(12, 6))
    contingency_table.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
    ax.set_title(f'Date Categorical Distribution (Stacked Bar): {date_col} vs {cat_feature}\n(p-value: {p_value:.4f})')
    ax.set_xlabel('Date (Monthly)')
    ax.set_ylabel('Count')
    ax.set_xticks(range(0, len(contingency_table), max(1, len(contingency_table) // 10)))
    ax.set_xticklabels([str(t) for t in contingency_table.index[::max(1, len(contingency_table) // 10)]], rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, plot_name)

def plot_non_linear(df, feature1, feature2, mutual_info, plot_name):
    # Plotting for non linear relation using scatter plot and density plot (that might show the actual connection)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=df[feature1], y=df[feature2], ax=axs[0])
    axs[0].set_title('Scatter Plot')
    sns.kdeplot(
        x=df[feature1], 
        y=df[feature2], 
        cmap="YlGnBu", 
        shade=True, 
        ax=axs[1]
    )
    axs[1].set_title(f'Density Plot (MI: {mutual_info:.2f})')
    plt.tight_layout()
    save_plot(fig, plot_name)

def plot_feature_importance(df, feature_details, target_variable, plot_name):
    # Plotting for feature importance using bar plot and cumulative importance.
    importances = {feature: info['importance_value'] for feature, info in feature_details.items()}
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.barplot(x=list(importances.keys()), y=list(importances.values()), ax=axs[0])
    axs[0].set_title(f'Feature Importances for {target_variable}')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
    
    cumulative_importance = np.cumsum(sorted(importances.values(), reverse=True))
    axs[1].plot(range(1, len(cumulative_importance) + 1), cumulative_importance)
    axs[1].set_title('Cumulative Feature Importance')
    axs[1].set_xlabel('Number of Features')
    axs[1].set_ylabel('Cumulative Importance')
    
    plt.tight_layout()
    save_plot(fig, plot_name)


def plot_outlier_pattern(df, feature1, feature2, plot_name):
    # Plotting outlier patter between 2 numerical features showing the outliers highlighted
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(x=feature1, y=feature2, data=df, ax=axs[0])
    axs[0].set_title('Full Data Scatter')
    z_scores_1 = np.abs((df[feature1] - df[feature1].mean()) / df[feature1].std())
    z_scores_2 = np.abs((df[feature2] - df[feature2].mean()) / df[feature2].std())
    outliers = df[(z_scores_1 > 3) | (z_scores_2 > 3)]
    sns.scatterplot(x=feature1, y=feature2, data=df, ax=axs[1], alpha=0.3)
    sns.scatterplot(x=feature1, y=feature2, data=outliers, color='red', ax=axs[1])
    axs[1].set_title('Outliers Highlighted')
    plt.tight_layout()
    save_plot(fig, plot_name)

def plot_target_analysis(df, target_variable, outlier_ratio, distribution_type, plot_name):
    # Plotting the target analysis using histogram plot and with the closest distribution
    fig, ax = plt.subplots()
    sns.histplot(df[target_variable], kde=True, ax=ax)
    ax.set_title(f'Target Analysis: {target_variable} (Outlier Ratio: {outlier_ratio:.2f}, Best Fit: {distribution_type})')
    save_plot(fig, plot_name)
