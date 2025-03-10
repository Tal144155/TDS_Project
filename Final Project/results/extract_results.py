import re
import os
import pandas as pd


# Function to extract ratings from text
def extract_ratings(text, pattern, name):
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Group ratings by type
    system_ratings = []
    random_ratings = []
    print (matches)
    for type, _ ,rating, _, comment, time in matches:
        user_dict = {'Type': type, 'Rating': int(rating), 'Comment': comment, 'Time': float(time) }
        if type == "System":
            system_ratings.append(user_dict)
        elif type == "Random":
            random_ratings.append(user_dict)
    
    return {
        "System": system_ratings,
        "Random": random_ratings
    }

base_path = './Final Project/results'
folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
# print(folders)
# Regex pattern to extract Type (Random / ours), Rating, the comment and the time taken to submit in seconds.
pattern = r'Type:\s+(System|Random)\s+(.*\s+)Rating:\s+(\d+)(.*\s+)Comment:\s+(.*\s+)Time Taken:\s+(\d+.\d+)'
df = pd.DataFrame({})

# for each person part the information to system and random
for folder in folders:
    file = open(base_path+'/'+folder+'/'+'user_feedback.txt', "r")

    # Get the ratings
    ratings = extract_ratings(file.read(), folder)
    # Print results
    # print("System ratings:", ratings["System"])
    # print("Random ratings:", ratings["Random"])
    print(f"System average: {sum(ratings['System']['Rating'])/len(ratings['System']) if ratings['System']['Rating'] else 0}")
    print(f"Random average: {sum(ratings['Random']['Rating'])/len(ratings['Random']) if ratings['Random']['Rating'] else 0}")