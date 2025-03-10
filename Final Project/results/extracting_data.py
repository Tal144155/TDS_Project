import os
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
                    name_in_dir = name_in_dir_line.replace('Name in Dir:  plot_', '')
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
    # Bar Chart: Average ratings for System vs. Random plots
    plt.figure(figsize=(8, 6))
    avg_rating_by_type = df.groupby('Type')['Rating'].mean().reset_index()
    sns.barplot(x='Type', y='Rating', hue='Type', data=avg_rating_by_type, palette='coolwarm', legend=False)
    plt.title('Average Ratings for System vs. Random Plots')
    plt.xlabel('Plot Type')
    plt.ylabel('Average Rating')
    plt.savefig(f'{output_path}/avg_rating_bar_chart.png')

    # Box Plot: Distribution of ratings across different relation types
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Relation Type', y='Rating', hue='Relation Type', data=df, palette='Set2', legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Ratings by Relation Type')
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
        palette='Set1',
        width=0.4
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Ratings by Relation Type (System vs. Random) - Filtered')
    plt.xlabel('Relation Type')
    plt.ylabel('Average Rating')
    plt.tight_layout()
    plt.savefig(f'{output_path}/filtered_avg_rating_by_relation_type_bar_chart.png')


base_path = r'C:\year3\TDS_Project\Final Project\results'
output = r'C:\year3\TDS_Project\Final Project\results\stats'

df = create_feedback_dataframe(base_path)
avg_rating_by_type, avg_rating_by_relation_type, avg_time_by_type, avg_rating_by_relation_combined = calculate_and_save_averages(df, base_path)
create_visualizations(df, avg_rating_by_relation_combined, base_path)


