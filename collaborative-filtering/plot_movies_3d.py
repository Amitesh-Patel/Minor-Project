import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Convert 'genres' list to a single string for visualization
movies["genres"] = movies["genres"].apply(
    lambda x: ", ".join(x) if x else "Unknown"
)

# Create content string
movies["content"] = (
    movies["cast"]
    + movies["crew"]
    + movies["genres"].apply(lambda g: g.split(", "))
    + movies["keywords"]
)
movies["content"] = movies["content"].apply(lambda x: " ".join(x).lower())

# Generate similarity matrix
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["content"]).toarray()
similarity_matrix = cosine_similarity(vectors)

# Dimensionality reduction using PCA
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(similarity_matrix)

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Scatter points
x = reduced_data[:, 0]
y = reduced_data[:, 1]
z = reduced_data[:, 2]
colors = np.random.rand(len(movies))  # Random colors for each movie

scatter = ax.scatter(x, y, z, c=colors, cmap="viridis", s=10, alpha=0.8)

# Set axis labels
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
plt.title("3D Visualization of Movies Based on Similarity")

# Optional: annotate a few movies for clarity
for i, title in enumerate(movies["title"].head(10)):
    ax.text(x[i], y[i], z[i], title, size=8)

plt.show()
