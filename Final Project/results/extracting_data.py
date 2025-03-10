import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def create_feedback_dataframe(base_path):
    data = []
    user_number = 1
    
    # Iterate through each folder (representing a user)
    for user_folder in os.listdir(base_path):
        user_folder_path = os.path.join(base_path, user_folder)
        if os.path.isdir(user_folder_path):
            feedback_file = os.path.join(user_folder_path, 'user_feedback.txt')
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_lines = f.readlines()
                
                # Parse the feedback lines
                for i in range(1, len(feedback_lines), 7):
                    plot_name_line = feedback_lines[i].strip()
                    type_line = feedback_lines[i+1].strip()
                    name_in_dir_line = feedback_lines[i+2].strip()
                    rating_line = feedback_lines[i+3].strip()
                    comment_line = feedback_lines[i+4].strip()
                    time_taken_line = feedback_lines[i+5].strip()

                    plot_name = plot_name_line.replace('Plot Name: ', '')
                    plot_type = type_line.replace('Type: ', '')
                    name_in_dir = name_in_dir_line.replace('Name in Dir: plot_', '')
                    rating = int(rating_line.replace('Rating: ', ''))
                    comment = comment_line.replace('Comment: ', '')
                    time_taken = float(time_taken_line.replace('Time Taken: ', '').replace(' seconds', ''))
                    
                    # Extracting the relation type (first part of the plot name)
                    relation_type = plot_name.split(' ')[0] if ' ' in plot_name else plot_name
                    relation = plot_name.split('between')[1].strip() if 'between' in plot_name else ''

                    data.append({
                        'User Number': user_number,
                        'User Name': user_folder,
                        'Plot Number': name_in_dir,
                        'Type': plot_type,
                        'Plot Name': plot_name,
                        'Relation Type': relation_type,
                        'Relation': relation,
                        'Rating': rating,
                        'Comment': comment,
                        'Time Taken': time_taken
                    })
            user_number += 1

    df = pd.DataFrame(data)
    return df

def calculate_and_save_averages(df, output_path):
    # Calculate the average rating for System and Random plots
    avg_rating_by_type = df.groupby('Type')['Rating'].mean().reset_index()
    avg_rating_by_type.columns = ['Plot Type', 'Average Rating']
    
    # Calculate the average rating per relation type only for "System" plots
    system_df = df[df['Type'] == 'System']
    avg_rating_by_relation_type = system_df.groupby('Relation Type')['Rating'].mean().reset_index()
    avg_rating_by_relation_type.columns = ['Relation Type', 'System Average Rating']
    
    # Calculate the average rating per relation type only for "Random" plots
    random_df = df[df['Type'] == 'Random']
    avg_rating_by_relation_random = random_df.groupby('Relation Type')['Rating'].mean().reset_index()
    avg_rating_by_relation_random.columns = ['Relation Type', 'Random Average Rating']
    
    # Merge the two results to compare system and random averages per relation type
    avg_rating_by_relation_combined = pd.merge(
        avg_rating_by_relation_type, 
        avg_rating_by_relation_random, 
        on='Relation Type', 
        how='outer'
    ).fillna(0)

    # Calculate the average time taken to rate each plot type
    avg_time_by_type = df.groupby('Type')['Time Taken'].mean().reset_index()
    avg_time_by_type.columns = ['Plot Type', 'Average Time Taken (seconds)']
    
    # Save the results as CSV files
    avg_rating_by_type.to_csv(f'{output_path}/avg_rating_by_type.csv', index=False, sep='\t')
    avg_rating_by_relation_type.to_csv(f'{output_path}/avg_rating_by_relation_type.csv', index=False, sep='\t')
    avg_rating_by_relation_combined.to_csv(f'{output_path}/avg_rating_by_relation_combined.csv', index=False, sep='\t')
    avg_time_by_type.to_csv(f'{output_path}/avg_time_by_type.csv', index=False, sep='\t')
    
    print("Results saved to:")
    print(f"{output_path}/avg_rating_by_type.csv")
    print(f"{output_path}/avg_rating_by_relation_type.csv")
    print(f"{output_path}/avg_time_by_type.csv")
    
    return avg_rating_by_type, avg_rating_by_relation_type, avg_time_by_type, avg_rating_by_relation_combined

def create_visualizations(df, avg_rating_by_relation_combined, output_path):
    custom_palette = sns.color_palette("Blues_d", as_cmap=False)
    selected = [custom_palette[0], custom_palette[4]]

    # Bar Chart: Average ratings for System vs. Random plots
    plt.figure(figsize=(8, 6))
    avg_rating_by_type = df.groupby('Type')['Rating'].mean().reset_index()
    sns.barplot(x='Type', y='Rating', hue='Type', data=avg_rating_by_type, palette=selected, legend=False)
    plt.title('Average Ratings for System vs. Random Plots')
    plt.xlabel('Plot Type')
    plt.ylabel('Average Rating')
    plt.savefig(f'{output_path}/avg_rating_bar_chart.png')

    # Box Plot: Distribution of ratings across different relation types
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Relation Type', y='Rating', hue='Relation Type', data=df, palette=custom_palette, legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Ratings by Relation Type for System Plots')
    plt.xlabel('Relation Type')
    plt.ylabel('Rating')
    plt.tight_layout()
    plt.savefig(f'{output_path}/rating_distribution_box_plot.png')

    # Filter out relation types where either System or Random average is 0
    filtered_avg_rating = avg_rating_by_relation_combined[
        (avg_rating_by_relation_combined['System Average Rating'] > 0) &
        (avg_rating_by_relation_combined['Random Average Rating'] > 0)
    ]
    
    # Melt the filtered DataFrame for visualization
    melted_df = filtered_avg_rating.melt(
        id_vars='Relation Type',
        value_vars=['System Average Rating', 'Random Average Rating'],
        var_name='Plot Type',
        value_name='Average Rating'
    )
    
    # Bar Graph: Average Ratings by Relation Type for System and Random Plots
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x='Relation Type',
        y='Average Rating',
        hue='Plot Type',
        data=melted_df,
        palette=selected,
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Ratings by Relation Type (System vs. Random)')
    plt.xlabel('Relation Type')
    plt.ylabel('Average Rating')
    plt.tight_layout()
    plt.savefig(f'{output_path}/filtered_avg_rating_by_relation_type_bar_chart.png')


    # Plot histogram: Histogram of Ratings Count for System and Random
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Rating", hue="Type", multiple="stack", bins=5, edgecolor="white" ,palette=selected)
    plt.xlabel("Rating")
    plt.ylabel("Count of Items")
    plt.title("Histogram of Ratings Count (System vs. Random)")
    plt.savefig(f'{output_path}/Histogram_of_Ratings_by_Type.png')

    # Average Time Taken vs. Plot Number for System and Random
    df_selected = df[["Plot Number", "Time Taken", "Type"]]
    df_avg = df_selected.groupby(["Plot Number", "Type"]).mean().reset_index()
    df_system = df_avg[df_avg["Type"] == "System"]
    df_random = df_avg[df_avg["Type"] == "Random"]
    df_system["Plot Number"] = range(len(df_system))  # Start from 0 for System
    df_random["Plot Number"] = range(len(df_random))  # Continue for Random

    plt.figure(figsize=(12, 6))
    plt.plot(df_system["Plot Number"], df_system["Time Taken"], marker='o', linestyle='-', label="System", color=custom_palette[0])
    plt.plot(df_random["Plot Number"], df_random["Time Taken"], marker='s', linestyle='-', label="Random", color=custom_palette[4])
    plt.xlabel("Plot Number")
    plt.ylabel("Average Time Taken")
    plt.title("Average Time Taken vs. Plot Number (System vs. Random)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/Average_Time_Taken_vs_Plot_Number.png')

def analyze_statistical_significance(df):
    # Separate ratings by System and Random types
    system_ratings = df[df['Type'] == 'System']['Rating']
    random_ratings = df[df['Type'] == 'Random']['Rating']
    
    # Conduct an independent t-test
    t_stat, p_value = stats.ttest_ind(system_ratings, random_ratings, equal_var=False)
    
    print(f"T-Test Results:")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    
    # Perform ANOVA on ratings by Relation Type
    relation_groups = [df[df['Relation Type'] == rt]['Rating'] for rt in df['Relation Type'].unique()]
    f_stat, p_anova = stats.f_oneway(*relation_groups)
    
    print(f"\nANOVA Results:")
    print(f"F-statistic: {f_stat:.4f}, p-value: {p_anova:.4f}")
    
    # Check if certain relation types consistently received higher or lower ratings
    relation_rating_summary = df.groupby('Relation Type')['Rating'].describe()
    print("\nRating Summary by Relation Type:")
    print(relation_rating_summary)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

def post_hoc_analysis(df, output_path):
    # Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(endog=df['Rating'], groups=df['Relation Type'], alpha=0.05)
    print("\nPost Hoc Analysis (Tukey's HSD):")
    print(tukey_results)
    
    # Plot the Tukey HSD results
    plt.figure(figsize=(10, 6))
    tukey_results.plot_simultaneous()
    plt.title('Tukey HSD Test: Comparison of Ratings by Relation Type')
    plt.xlabel('Mean Rating Difference')
    plt.ylabel('Relation Type')
    plt.tight_layout()
    plt.savefig(f'{output_path}/tukey_hsd_relation_type_comparison.png')

base_path = r'C:\year3\TDS_Project\Final Project\results'
output = r'C:\year3\TDS_Project\Final Project\results\stats'
os.makedirs(output, exist_ok=True)
df = create_feedback_dataframe(base_path)
avg_rating_by_type, avg_rating_by_relation_type, avg_time_by_type, avg_rating_by_relation_combined = calculate_and_save_averages(df, output)
create_visualizations(df, avg_rating_by_relation_combined, output)
analyze_statistical_significance(df)
post_hoc_analysis(df, output)


