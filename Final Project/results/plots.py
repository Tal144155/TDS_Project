import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from extraction_functions import *

my_base_path = './Final Project/results'

df = create_feedback_dataframe(my_base_path)

# avg_rating_by_type, avg_rating_by_relation_type, avg_time_by_type = calculate_and_save_averages(df, my_base_path)

sns.set_theme(style="whitegrid")

# Plot histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Rating", hue="Type", multiple="stack", bins=5, palette="viridis")

# Labels and title
plt.xlabel("Rating")
plt.ylabel("Count of Items")
plt.title("Histogram of Ratings by Type")

# Show the plot
plt.show()

df_selected = df[["Plot Number", "Time Taken", "Type"]]

df_avg = df_selected.groupby(["Plot Number", "Type"]).mean().reset_index()

# Separate data based on "Type"
df_system = df_avg[df_avg["Type"] == "System"]
df_random = df_avg[df_avg["Type"] == "Random"]

# Adjust plot numbers to visually separate System (left) and Random (right)
df_system["Plot Number"] = range(len(df_system))  # Start from 0 for System
df_random["Plot Number"] = range(len(df_random))  # Continue for Random

# Create the line plots
plt.figure(figsize=(12, 6))

# Plot System on the left
plt.plot(df_system["Plot Number"], df_system["Time Taken"], marker='o', linestyle='-', label="System", color='blue')

# Plot Random on the right
plt.plot(df_random["Plot Number"], df_random["Time Taken"], marker='s', linestyle='-', label="Random", color='red')

# Labels and title
plt.xlabel("Plot Number")
plt.ylabel("Average Time Taken")
plt.title("Average Time Taken vs. Plot Number (Separated by Type)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()