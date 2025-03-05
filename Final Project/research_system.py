import os
import time
import random
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import scrolledtext
from plot_generator import *  # Assuming plot_generator.py is in the same directory
from recommendation_tool import *  # Assuming your recommendation code is in this module

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

# Frame to display plot images
plot_frame = tk.Frame(app)
plot_frame.pack(pady=20)

# Label to show plot information
plot_label = tk.Label(plot_frame, text="", font=("Arial", 14))
plot_label.pack()

# Text box for user comments
comment_box = scrolledtext.ScrolledText(app, wrap=tk.WORD, height=5)
comment_box.pack(pady=10)

# Entry for rating
rating_var = tk.StringVar()
rating_label = tk.Label(app, text="Rate this plot (1-5):")
rating_label.pack()
rating_entry = tk.Entry(app, textvariable=rating_var)
rating_entry.pack()

# Button to submit feedback
def submit_feedback():
    global plot_index, plot_data, start_time
    rating = int(rating_var.get()) if rating_var.get().isdigit() else None
    comment = comment_box.get("1.0", tk.END).strip()
    if not rating or rating < 1 or rating > 5:
        messagebox.showerror("Invalid Input", "Please provide a rating between 1 and 5.")
        return
    if not comment:
        messagebox.showerror("Invalid Input", "Please provide a comment.")
        return
    end_time = time.time()
    elapsed_time = end_time - start_time
    plot_info = plot_data[plot_index]
    plot_info.update({
        'rating': rating,
        'comment': comment,
        'time_taken': elapsed_time
    })
    plot_index += 1
    if plot_index < len(plot_data):
        show_next_plot()
    else:
        save_results()
        messagebox.showinfo("Process Complete", "Thank you! The process is complete.")
        app.quit()

submit_button = tk.Button(app, text="Submit Feedback", command=submit_feedback)
submit_button.pack(pady=20)

# Function to start the testing process
def start_process():
    global start_time, plot_data, user_id, ratings, plot_index
    user_id = simpledialog.askstring("User ID", "Please enter your User ID:")
    if not user_id:
        messagebox.showerror("Invalid Input", "User ID is required to start.")
        return
    ratings = load_ratings('user_ratings_rel', RELATION_TYPES)
    if user_id not in ratings.index:
        ratings.loc[user_id] = np.nan
        save_ratings(ratings, 'user_ratings_rel')
    plot_data = generate_plots()  # Implement this function to generate plot data
    plot_index = 0
    show_next_plot()

# Function to display the next plot
def show_next_plot():
    global start_time
    plot_info = plot_data[plot_index]
    plot_label.config(text=f"Plot {plot_info['name']} ({'System' if plot_info['is_system'] else 'Random'})")
    comment_box.delete("1.0", tk.END)
    rating_var.set("")
    start_time = time.time()

# Function to save results to a text file
def save_results():
    results_file = "user_feedback.txt"
    with open(results_file, 'w') as f:
        for plot in plot_data:
            f.write(f"Plot Name: {plot['name']}
")
            f.write(f"Type: {'System' if plot['is_system'] else 'Random'}
")
            f.write(f"Rating: {plot['rating']}
")
            f.write(f"Comment: {plot['comment']}
")
            f.write(f"Time Taken: {plot['time_taken']:.2f} seconds

")
    print(f"Results saved to {results_file}")

# Button to start the process
start_button = tk.Button(app, text="Start", command=start_process)
start_button.pack(pady=50)

app.mainloop()
