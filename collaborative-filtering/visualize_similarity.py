import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load datasets
movies_path = r"C:\Users\mrami\OneDrive\Desktop\Minor_project\dataset\tmdb_5000_movies.csv"
credits_path = r"C:\Users\mrami\OneDrive\Desktop\Minor_project\dataset\tmdb_5000_credits.csv"

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)


# Preprocessing and merging datasets
def convert_json_to_list(json_str, key="name", max_items=None):
    """Convert JSON string to list of names"""
    items = json.loads(json_str)
    if max_items:
        items = items[:max_items]
    return [item[key] for item in items]


def get_director(crew_json):
    """Extract director name from crew JSON"""
    crew = json.loads(crew_json)
    for member in crew:
        if member["job"] == "Director":
            return [member["name"]]
    return []


movies = movies.merge(credits, on="title")
columns = [
    "movie_id",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew",
]
movies = movies[columns].dropna()

movies["genres"] = movies["genres"].apply(lambda x: convert_json_to_list(x))
movies["keywords"] = movies["keywords"].apply(
    lambda x: convert_json_to_list(x)
)
movies["cast"] = movies["cast"].apply(
    lambda x: convert_json_to_list(x, max_items=3)
)
movies["crew"] = movies["crew"].apply(get_director)

# Remove spaces from strings
for column in ["cast", "crew", "genres", "keywords"]:
    movies[column] = movies[column].apply(
        lambda x: [item.replace(" ", "") for item in x]
    )

# Create content string
movies["content"] = (
    movies["cast"] + movies["crew"] + movies["genres"] + movies["keywords"]
)
movies["content"] = movies["content"].apply(lambda x: " ".join(x).lower())

# Generate similarity matrix
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["content"]).toarray()
similarity_matrix = cosine_similarity(vectors)


# Visualize similarity matrix
def plot_similarity_matrix(matrix, labels):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix[:20, :20],  # Showing only a small portion for clarity
        cmap="viridis",
        xticklabels=labels[:20],
        yticklabels=labels[:20],
        annot=False,
        fmt=".2f",
    )
    plt.title("Movie Similarity Matrix (Partial View)")
    plt.xlabel("Movies")
    plt.ylabel("Movies")
    plt.tight_layout()
    plt.show()


movie_titles = movies["title"].tolist()
plot_similarity_matrix(similarity_matrix, movie_titles)
