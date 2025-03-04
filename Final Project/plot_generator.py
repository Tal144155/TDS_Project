import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def save_plot(fig, plot_name):
    file_path = os.path.join(PLOTS_DIR, f"{plot_name}.png")
    fig.savefig(file_path)
    plt.close(fig)
    print(f"Saved plot: {file_path}")

def plot_high_correlation(df, feature1, feature2, correlation_value):
    fig, ax = plt.subplots()
    sns.regplot(x=df[feature1], y=df[feature2], scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'color': 'red'})
    ax.set_title(f'High Correlation: {feature1} vs {feature2} (Corr: {correlation_value:.2f})')
    save_plot(fig, f'high_correlation_{feature1}_{feature2}')

def plot_target_correlation(df, feature, target, correlation_value):
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature], y=df[target], ax=ax)
    ax.set_title(f'Target Correlation: {feature} vs {target} (Corr: {correlation_value:.2f})')
    save_plot(fig, f'target_correlation_{feature}_{target}')

def plot_categorical_effect(df, categorical_feature, target, p_value):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[categorical_feature], y=df[target], ax=ax)
    ax.set_title(f'Categorical Effect: {categorical_feature} on {target} (p-value: {p_value:.4f})')
    save_plot(fig, f'categorical_effect_{categorical_feature}_{target}')

def plot_chi_squared(df, feature1, feature2, p_value):
    contingency_table = pd.crosstab(df[feature1], df[feature2])
    fig, ax = plt.subplots()
    sns.heatmap(contingency_table, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title(f'Chi-Squared Test: {feature1} vs {feature2} (p-value: {p_value:.4f})')
    save_plot(fig, f'chi_squared_{feature1}_{feature2}')

def plot_date_numerical_trend(df, date_col, num_feature, correlation_value):
    fig, ax = plt.subplots()
    sns.lineplot(x=df[date_col], y=df[num_feature], ax=ax)
    ax.set_title(f'Date Numerical Trend: {date_col} vs {num_feature} (Corr: {correlation_value:.2f})')
    save_plot(fig, f'date_numerical_trend_{date_col}_{num_feature}')

def plot_date_categorical_distribution(df, date_col, cat_feature, p_value):
    contingency_table = pd.crosstab(df[date_col].dt.to_period('M'), df[cat_feature])
    fig, ax = plt.subplots(figsize=(10, 6))
    contingency_table.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
    ax.set_title(f'Date Categorical Distribution (Stacked Bar): {date_col} vs {cat_feature}\n(p-value: {p_value:.4f})')
    ax.set_xlabel('Date (Monthly)')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, f'date_categorical_distribution_{date_col}_{cat_feature}')

def plot_non_linear(df, feature1, feature2, mutual_info):
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature1], y=df[feature2], ax=ax)
    ax.set_title(f'Non-Linear Relationship: {feature1} vs {feature2} (MI: {mutual_info:.2f})')
    save_plot(fig, f'non_linear_{feature1}_{feature2}')

def plot_feature_importance(df, feature, target, importance_value):
    fig, ax = plt.subplots()
    sns.barplot(x=[feature], y=[importance_value], ax=ax)
    ax.set_title(f'Feature Importance: {feature} for {target} (Importance: {importance_value:.2f})')
    save_plot(fig, f'feature_importance_{feature}_{target}')

def plot_outlier_pattern(df, feature1, feature2, outlier_count):
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature1], y=df[feature2], ax=ax)
    ax.set_title(f'Outlier Pattern: {feature1} vs {feature2} (Outliers: {outlier_count})')
    save_plot(fig, f'outlier_pattern_{feature1}_{feature2}')

def plot_cluster_group(df, selected_features, cluster_id):
    fig, ax = plt.subplots()
    sns.pairplot(df[selected_features], hue=cluster_id)
    plt.suptitle(f'Cluster Group: {cluster_id}', y=1.02)
    save_plot(fig, f'cluster_group_{cluster_id}')

def plot_target_analysis(df, target_variable, outlier_ratio, distribution_type, distribution_p_value):
    fig, ax = plt.subplots()
    sns.histplot(df[target_variable], kde=True, ax=ax)
    ax.set_title(f'Target Analysis: {target_variable} (Outlier Ratio: {outlier_ratio:.2f}, Best Fit: {distribution_type})')
    save_plot(fig, f'target_analysis_{target_variable}')
