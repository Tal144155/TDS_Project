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

# Function to switch from the opening screen to the main process
def switch_to_main():
    opening_frame.pack_forget()
    main_frame.pack(expand=True)
    start_process()

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
    user_rating = 0
    if pd.notna(ratings.loc[user_id, chosen_plot['relation_type']]):
        user_rating = ratings.loc[user_id, chosen_plot['relation_type']]
    if user_rating:
        ratings.loc[user_id, chosen_plot['relation_type']] = user_rating * 0.8 + rating * 0.2
    else:
        ratings.loc[user_id, chosen_plot['relation_type']] = rating

    save_ratings(ratings, 'user_ratings_rel') 
    end_time = time.time()
    elapsed_time = end_time - start_time
    plot_info = plot_data[plot_index]
    plot_info.update({
        'rating': rating,
        'comment': comment,
        'time_taken': elapsed_time
    })
    ratings = load_ratings('user_ratings_rel', RELATION_TYPES)
    plot_index += 1
    if algo_rec:
        show_next_plot()
    else:
        save_results()
        messagebox.showinfo("Process Complete", "Thank you! The process is complete.")
        app.quit()

# Function to start the testing process
def start_process():
    global start_time, user_id, ratings, plot_index, algo_rec, df
    user_id = simpledialog.askstring("User ID", "Please enter your User ID:")
    if not user_id:
        messagebox.showerror("Invalid Input", "User ID is required to start.")
        return
    ratings = load_ratings('user_ratings_rel', RELATION_TYPES)
    if user_id not in ratings.index:
        ratings.loc[user_id] = np.nan
        save_ratings(ratings, 'user_ratings_rel')
    plot_index = 0
    dataset_path = "Final Project/Datasets_Testing/AB_NYC_2019.csv"
    index_col = "id"
    target_value = "price"
    df = read_data(dataset_path, index_col)
    if df is None:
        return
    dataset_types = get_column_types(df)
    algo_rec = find_relations(df, target_value, dataset_types)
    algo_rec = get_relation_scores(algo_rec)
    show_next_plot()

# Function to generate a new plot dynamically
def generate_plot():
    global user_id, algo_rec, chosen_plot
    is_system_plot = random.choice([True, True])
    plot_name = f'plot{plot_index+1}'
    image_path = ""
    ratings = load_ratings('user_ratings_rel', RELATION_TYPES)
    if is_system_plot and algo_rec:
        combined_user_vis_pred = combine_pred(CFCB(ratings), CFUB(ratings), 0.5, 0.5)
        algo_rec_df = get_top_relations(algo_rec)
        user_index = ratings.index.get_loc(user_id)
        recommendations = combine_pred(combined_user_vis_pred[user_index], algo_rec_df.to_numpy()[0], 0.7, 0.3)
        index = int(algo_rec_df.iloc[1,recommendations.argmax()])
        chosen_plot = algo_rec.pop(index)
        plot_name = f'system_{chosen_plot["relation_type"]}_{plot_index+1}'
        plot_save_name = f'plot_{plot_index}'
        plot_info = {
            'name': plot_name,
            'is_system': True,
            'relation_type': chosen_plot['relation_type'],
            'attributes': chosen_plot["attributes"],
            'plot_save_name': plot_save_name
        }
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
            plot_feature_importance()
        elif chosen_plot['relation_type'] == "outlier_pattern":
            plot_outlier_pattern(df, chosen_plot["attributes"][0], chosen_plot["attributes"][1], plot_save_name)
        elif chosen_plot['relation_type'] == "cluster_group":
            plot_cluster_group()
        elif chosen_plot['relation_type'] == "target_analysis":
            plot_target_analysis(df, chosen_plot["attributes"][0], chosen_plot["details"]["outlier_ratio"], chosen_plot["details"]["distribution_type"], plot_save_name)

        image_path = os.path.abspath(os.path.join(PLOTS_DIR, f'{plot_save_name}.png'))
    else:
        features = random.sample(df.columns.tolist(), 2)
        plot_name = f'random_{features[0]}_{features[1]}_{plot_index+1}'
        plot_info = {
            'name': plot_name,
            'is_system': False
        }
        plot_high_correlation(df, features[0], features[1], random.uniform(0.5, 1.0))
        image_path = os.path.join(PLOTS_DIR, f'{plot_save_name}.png')
    plot_data.append(plot_info)
    display_plot(image_path)
    return plot_info


def display_plot(image_path):
    if not os.path.exists(image_path):
        messagebox.showerror("Error", f"Image not found: {image_path}")
        return

    try:
        # Open the image
        image = Image.open(image_path)

        # Define the maximum size
        max_width, max_height = 600, 400

        # Maintain aspect ratio using thumbnail
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage for tkinter
        photo = ImageTk.PhotoImage(image)

        # Update the plot_canvas
        plot_canvas.config(image=photo)
        plot_canvas.image = photo  # Keep a reference to avoid garbage collection

        print(f"Displayed image: {image_path}")

    except Exception as e:
        print(f"Failed to load image: {e}")
        messagebox.showerror("Error", f"Failed to load image: {e}")


# Function to display the next plot
def show_next_plot():
    global start_time
    plot_info = generate_plot()
    plot_label.config(text=f"Plot {plot_info['name']} ({'System' if plot_info['is_system'] else 'Random'})")
    comment_box.delete("1.0", tk.END)
    rating_var.set("")
    time.sleep(0.1)
    start_time = time.time()

# Function to save results to a text file
def save_results():
    results_file = "user_feedback.txt"
    with open(results_file, 'w') as f:
        for plot in plot_data:
            f.write(f"Plot Name: {plot['name']}\n")
            f.write(f"Type: {'System' if plot['is_system'] else 'Random'}\n")
            f.write(f"Name in Dir: {plot['plot_save_name']}\n")
            f.write(f"Rating: {plot.get('rating', 'N/A')}\n")
            f.write(f"Comment: {plot.get('comment', 'N/A')}\n")
            f.write(f"Time Taken: {plot.get('time_taken', 0):.2f} seconds\n\n")
    print(f"Results saved to {results_file}")

app.mainloop()