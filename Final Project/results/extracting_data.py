import os
import pandas as pd

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
                    name_in_dir = name_in_dir_line.replace('Name in Dir: ', '')
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
    
    # Calculate the average rating per relation type
    system_df = df[df['Type'] == 'System']
    
    # Calculate the average rating per relation type only for "System" plots
    avg_rating_by_relation_type = system_df.groupby('Relation Type')['Rating'].mean().reset_index()
    avg_rating_by_relation_type.columns = ['Relation Type', 'Average Rating']
    
    # Calculate the average time taken to rate each plot type
    avg_time_by_type = df.groupby('Type')['Time Taken'].mean().reset_index()
    avg_time_by_type.columns = ['Plot Type', 'Average Time Taken (seconds)']
    
    # Save the results as CSV files
    avg_rating_by_type.to_csv(f'{output_path}/avg_rating_by_type.csv', index=False, sep='\t')
    avg_rating_by_relation_type.to_csv(f'{output_path}/avg_rating_by_relation_type.csv', index=False, sep='\t')
    avg_time_by_type.to_csv(f'{output_path}/avg_time_by_type.csv', index=False, sep='\t')
    
    print("Results saved to:")
    print(f"{output_path}/avg_rating_by_type.csv")
    print(f"{output_path}/avg_rating_by_relation_type.csv")
    print(f"{output_path}/avg_time_by_type.csv")
    
    return avg_rating_by_type, avg_rating_by_relation_type, avg_time_by_type

base_path = r'C:\year3\TDS_Project\Final Project\results'
df = create_feedback_dataframe(base_path)

output_path = r'C:\year3\TDS_Project\Final Project\results'
avg_rating_by_type, avg_rating_by_relation_type, avg_time_by_type = calculate_and_save_averages(df, output_path)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def create_visualizations(df, output_path):
    # Bar Chart: Average ratings for System vs. Random plots
    plt.figure(figsize=(8, 6))
    avg_rating_by_type = df.groupby('Type')['Rating'].mean().reset_index()
    sns.barplot(x='Type', y='Rating', data=avg_rating_by_type, palette='viridis')
    plt.title('Average Ratings for System vs. Random Plots')
    plt.xlabel('Plot Type')
    plt.ylabel('Average Rating')
    plt.savefig(f'{output_path}/avg_rating_bar_chart.png')

    # Box Plot: Distribution of ratings across different relation types
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Relation Type', y='Rating', data=df, palette='muted')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Ratings by Relation Type')
    plt.xlabel('Relation Type')
    plt.ylabel('Rating')
    plt.tight_layout()
    plt.savefig(f'{output_path}/rating_distribution_box_plot.png')

    # Word Cloud: Comments visualization
    comments_text = ' '.join(df['Comment'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(comments_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Comments')
    plt.savefig(f'{output_path}/comments_word_cloud.png')

    # Heatmap: Ratings per user and relation type
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot_table(values='Rating', index='User Name', columns='Relation Type', aggfunc='mean')
    sns.heatmap(pivot_table, cmap='viridis', annot=True)
    plt.title('Heatmap of Ratings by User and Relation Type')
    plt.xlabel('Relation Type')
    plt.ylabel('User Name')
    plt.tight_layout()
    plt.savefig(f'{output_path}/ratings_heatmap.png')

# Example usage with the existing DataFrame and output path
output_path = r'C:\year3\TDS_Project\Final Project\results'
create_visualizations(df, output_path)


