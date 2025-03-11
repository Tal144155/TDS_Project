import os
import time
import random
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import scrolledtext
from PIL import Image, ImageTk
from plot_generator import *
from recommendation_tool import *
from relation_detection_algorithm import *

# Directory to save plots
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Variables to keep track of the process
plot_index = 0
plot_data = []
start_time = None
user_id = None
ratings = None

# Create the main application window
app = tk.Tk()
app.title("Recommendation System Testing")
app.geometry("800x600")

# Frame for the opening screen
opening_frame = tk.Frame(app)
opening_frame.pack(expand=True)

welcome_label = tk.Label(opening_frame, text="Welcome to the Recommendation System Testing", font=("Arial", 20))
welcome_label.pack(pady=20)

start_button = tk.Button(opening_frame, text="Start", font=("Arial", 16), command=lambda: switch_to_main())
start_button.pack(pady=50)

# Main UI elements (initially hidden)
main_frame = tk.Frame(app)
plot_label = tk.Label(main_frame, text="", font=("Arial", 14))
plot_label.pack(pady=10)

comment_box = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=5)
comment_box.pack(pady=10)

rating_var = tk.StringVar()
rating_label = tk.Label(main_frame, text="Rate this plot (1-5):")
rating_label.pack()
rating_entry = tk.Entry(main_frame, textvariable=rating_var)
rating_entry.pack()

submit_button = tk.Button(main_frame, text="Submit Feedback", command=lambda: submit_feedback())
submit_button.pack(pady=20)

plot_canvas = tk.Label(main_frame)
plot_canvas.pack(pady=10)

plot_order = []

# Function to switch from the opening screen to the main process
def switch_to_main():
    opening_frame.pack_forget()
    main_frame.pack(expand=True)
    start_process()

def start_process():
    # global variables needed
    global start_time,num_chose_outliers, user_id, ratings, plot_index, algo_rec, df, dataset_types, count_system, dataset_path, plot_order
    # asking for the user id to enter the system
    user_id = simpledialog.askstring("User ID", "Please enter your User ID:")
    if not user_id:
        messagebox.showerror("Invalid Input", "User ID is required to start.")
        return
    # load the ratings files
    ratings = load_ratings('user_ratings', RELATION_TYPES)
    if user_id not in ratings.index:
        ratings.loc[user_id] = np.nan
        save_ratings(ratings, 'user_ratings')
    plot_index = 0
    count_system = 0
    num_chose_outliers = 0
    dataset_path = "Final Project/Datasets_Testing/movie_new.csv"
    index_col = "id"
    target_value = "revenue"
    # upload the data and start the process
    df = read_data(dataset_path)
    if df is None:
        return
    # get the types and find the relations, then get the top ones
    dataset_types = get_column_types(df)
    algo_rec = find_relations(df, target_value, dataset_types)
    algo_rec = get_relation_scores(algo_rec)
    # define number of plots about to be shown
    num_system_plots = min(10, len(algo_rec))
    num_random_plots = 5
    # create an array with the order of displaying plots.
    plot_order = [True] * num_system_plots + [False] * num_random_plots
    # shuffle the array for random order
    random.shuffle(plot_order)

    show_next_plot()

def generate_plot():
    """This function is responsible for creating a new visualization, system or random,
    depends on the plot number"""
    global user_id, algo_rec, chosen_plot, is_system_plot, count_system, num_chose_outliers
    # if the plot is in the boundaries of the array, get it's type from the array
    if plot_index < len(plot_order):
        is_system_plot = plot_order[plot_index]
    else:
        is_system_plot = False
    # create the name of the plot
    plot_name = f'plot{plot_index+1}'
    # load the ratings
    ratings = load_ratings('user_ratings', RELATION_TYPES)
    # if it is a system plot
    if is_system_plot and algo_rec:
        # get the most wanted visualization by the user
        count_system = count_system + 1
        combined_user_vis_pred = combine_pred(CFIB(ratings), CFUB(ratings), 0.5, 0.5)
        algo_rec_df = get_top_relations(algo_rec)
        user_index = ratings.index.get_loc(user_id)
        recommendations = combine_pred(combined_user_vis_pred[user_index], algo_rec_df.to_numpy()[0], 0.7, 0.3)
        index = int(algo_rec_df.iloc[1,recommendations.argmax()])
        # take it out of the list
        chosen_plot = algo_rec.pop(index)
        # if the chosen plot is the outlier (which is shown a lot) we want it to be only twice.
        # if it is more then 2 times, choose another one instead
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
        # update the name of the plot with the number of features.
        if len(chosen_plot["attributes"]) == 1:
            plot_name = f'{chosen_plot["relation_type"]} between {chosen_plot["attributes"][0]}'
        else:
            plot_name = f'{chosen_plot["relation_type"]} between {chosen_plot["attributes"][0]}, {chosen_plot["attributes"][1]}'
        plot_save_name = f'plot_{plot_index}'
        # create the info of the plot with the important information
        plot_info = {
            'name': plot_name,
            'is_system': True,
            'relation_type': chosen_plot['relation_type'],
            'attributes': chosen_plot["attributes"],
            'plot_save_name': plot_save_name
        }
        # based on the relation type, call the appropriate method from plot generator
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
        # create the image path
        image_path = os.path.abspath(os.path.join(PLOTS_DIR, f'{plot_save_name}.png'))
    else:
        # otherwise, we want to create a random visualization, and search for 2 random features
        allowed_types = {"integer", "float", "datetime", "categorical_int", "categorical_string"}
        # run until you find 2 features from the list
        while True:
            selected_features = random.sample(list(dataset_types.keys()), 2)
            feature1, feature2 = selected_features[0], selected_features[1]
            feature1_type = dataset_types[feature1]
            feature2_type = dataset_types[feature2]
            
            if feature1_type in allowed_types and feature2_type in allowed_types:
                break
        # go over the same process as before
        plot_name = f'{feature1} {feature2}'
        plot_save_name = f'plot_{plot_index}'
        plot_info = {
            'is_system': False,
            'plot_save_name': plot_save_name,
            'attributes': [feature1, feature2]
        }
        # Handling different feature type combinations, there can be a lot of combinations and we want to handle them all
        if (feature1_type == "integer" or feature1_type == "float") and (feature2_type == "integer" or feature2_type == "float"):
            plot_type = random.choice(['high_correlation', 'non_linear'])
            if plot_type == "high_correlation":
                plot_high_correlation(df, feature1, feature2, random.uniform(0.5, 1.0), plot_save_name)
                plot_info["relation_type"] = "high_correlation"
            else:
                plot_non_linear(df, feature1, feature2, random.uniform(0.5, 1.0), plot_save_name)
                plot_info["relation_type"] = "non_linear"
        
        elif feature1_type == "datetime" and feature2_type == "integer" or feature2_type == "float":
            plot_date_numerical_trend(df, feature1, feature2, random.uniform(0.5, 1.0), plot_save_name)
            plot_info["relation_type"] = "date_numerical_trend"
        
        elif feature2_type == "datetime" and feature1_type == "integer" or feature1_type == "float":
            plot_date_numerical_trend(df, feature2, feature1, random.uniform(0.5, 1.0), plot_save_name)
            plot_info["relation_type"] = "date_numerical_trend"
        
        elif feature1_type == "datetime" and (feature2_type == "categorical_int" or feature2_type=="categorical_string"):
            plot_date_categorical_distribution(df, feature1, feature2, random.uniform(0, 0.01), plot_save_name)
            plot_info["relation_type"] = "date_categorical_distribution"
        
        elif feature2_type == "datetime" and (feature1_type == "categorical_int" or feature1_type=="categorical_string"):
            plot_date_categorical_distribution(df, feature2, feature1, random.uniform(0, 0.01), plot_save_name)
            plot_info["relation_type"] = "date_categorical_distribution"
        
        elif (feature1_type == "categorical_int" or feature1_type=="categorical_string") and (feature2_type == "integer" or feature2_type=="float"):
            plot_categorical_effect(df, feature1, feature2, random.uniform(0, 0.01), plot_save_name)
            plot_info["relation_type"] = "categorical_effect"
        
        elif (feature2_type == "categorical_int" or feature2_type=="categorical_string") and (feature1_type == "integer" or feature1_type=="float"):
            plot_categorical_effect(df, feature2, feature1, random.uniform(0, 0.01), plot_save_name)
            plot_info["relation_type"] = "categorical_effect"
        
        elif (feature1_type == "categorical_int" or feature1_type=="categorical_string") and (feature2_type == "categorical_int" or feature2_type=="categorical_string"):
            numerical_features = [col for col, typ in dataset_types.items() if typ == "integer" or typ == "float"]
            if numerical_features:
                numeric_feature = random.choice(numerical_features)
                plot_categorical_effect(df, feature1, numeric_feature, random.uniform(0, 0.01), plot_save_name)
                plot_info["relation_type"] = "categorical_effect"
            else:
                plot_chi_squared(df, feature1, feature2, random.uniform(0, 0.01), plot_save_name)
                plot_info["relation_type"] = "chi_squared"
        if len(plot_info["attributes"]) == 1:
            plot_info["name"] = f'{plot_info["relation_type"]} between {feature1}'
        else:
            plot_info["name"] = f'{plot_info["relation_type"]} between {feature1}, {feature2}'

        image_path = os.path.join(PLOTS_DIR, f'{plot_save_name}.png')
    # adding the info to the plot data
    plot_data.append(plot_info)
    # displaying the plot on the screen
    display_plot(image_path)
    return plot_info


def display_plot(image_path):
    # check that the image exists in the folder
    if not os.path.exists(image_path):
        messagebox.showerror("Error", f"Image not found: {image_path}")
        return

    try:
        # trying to present the plot for the user.
        image = Image.open(image_path)
        max_width, max_height = 600, 400
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        plot_canvas.config(image=photo)
        plot_canvas.image = photo

        print(f"Displayed image: {image_path}")

    except Exception as e:
        print(f"Failed to load image: {e}")
        messagebox.showerror("Error", f"Failed to load image: {e}")


# Function to display the next plot
def show_next_plot():
    # generate the plot for the user to see.
    global start_time
    plot_info = generate_plot()
    plot_label.config(text=f"Plot: {plot_info['name']}")
    comment_box.delete("1.0", tk.END)
    rating_var.set("")
    time.sleep(1)
    # start the time after the plot is shown
    start_time = time.time()

# Button to submit feedback
def submit_feedback():
    global plot_index, plot_data, start_time, ratings
    rating = int(rating_var.get()) if rating_var.get().isdigit() else None
    comment = comment_box.get("1.0", tk.END).strip()
    if not rating or rating < 1 or rating > 5:
        messagebox.showerror("Invalid Input", "Please provide a rating between 1 and 5.")
        return
    if not comment:
        messagebox.showerror("Invalid Input", "Please provide a comment.")
        return
    # only update the rating if its a system plot.
    if is_system_plot:
        user_rating = 0
        if pd.notna(ratings.loc[user_id, chosen_plot['relation_type']]):
            user_rating = ratings.loc[user_id, chosen_plot['relation_type']]
        if user_rating:
            # update with weight to the previews rating of this kind.
            ratings.loc[user_id, chosen_plot['relation_type']] = user_rating * 0.5 + rating * 0.5
        else:
            ratings.loc[user_id, chosen_plot['relation_type']] = rating
        save_ratings(ratings, 'user_ratings') 
    end_time = time.time()
    # calculate the time took to submit the rating and comments
    elapsed_time = end_time - start_time
    plot_info = plot_data[plot_index]
    # adding the information to the plot info
    plot_info.update({
        'rating': rating,
        'comment': comment,
        'time_taken': elapsed_time
    })
    # load the new ratings
    ratings = load_ratings('user_ratings', RELATION_TYPES)
    plot_index += 1
    # if we have more plots to show and we did not finish 15 plots, show a new one
    if algo_rec and plot_index < min(len(algo_rec), 10) + 5:
        show_next_plot()
    else:
        # save the results and end the process
        save_results()
        messagebox.showinfo("Process Complete", "Thank you! The process is complete.")
        app.quit()


# Function to save results to a text file
def save_results():
    # save the results after the user finished rating all the plots
    results_file = "user_feedback.txt"
    with open(results_file, 'w') as f:
        f.write(f"Results on test from dataset: {dataset_path}\n")
        for plot in plot_data:
            f.write(f"Plot Name: {plot['name']}\n")
            f.write(f"Type: {'System' if plot['is_system'] else 'Random'}\n")
            f.write(f"Name in Dir: {plot['plot_save_name']}\n")
            f.write(f"Rating: {plot.get('rating', 'N/A')}\n")
            f.write(f"Comment: {plot.get('comment', 'N/A')}\n")
            f.write(f"Time Taken: {plot.get('time_taken', 0):.2f} seconds\n\n")
    print(f"Results saved to {results_file}")

app.mainloop()