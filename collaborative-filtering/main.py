import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import json
import nltk
import plotly.express as px

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")


class MovieRecommender:
    def __init__(self, movies_path, credits_path):
        self.movies = pd.read_csv(movies_path)
        self.credits = pd.read_csv(credits_path)
        self.df = None
        self.similarity_matrix = None
        self.prepare_data()

    def convert_json_to_list(self, json_str, key="name", max_items=None):
        """Convert JSON string to list of names"""
        items = json.loads(json_str)
        if max_items:
            items = items[:max_items]
        return [item[key] for item in items]

    def get_director(self, crew_json):
        """Extract director name from crew JSON"""
        crew = json.loads(crew_json)
        for member in crew:
            if member["job"] == "Director":
                return [member["name"]]
        return []

    def prepare_data(self):
        """Prepare and clean the dataset"""
        # Merge datasets
        self.df = self.movies.merge(self.credits, on="title")

        columns = [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
            "vote_count",
            "vote_average",
            "popularity",
        ]
        self.df = self.df[columns]

        self.df.dropna(inplace=True)

        self.df["genres"] = self.df["genres"].apply(
            lambda x: self.convert_json_to_list(x)
        )
        self.df["keywords"] = self.df["keywords"].apply(
            lambda x: self.convert_json_to_list(x)
        )
        self.df["cast"] = self.df["cast"].apply(
            lambda x: self.convert_json_to_list(x, max_items=3)
        )
        self.df["crew"] = self.df["crew"].apply(self.get_director)

        # Remove spaces from strings
        for column in ["cast", "crew", "genres", "keywords"]:
            self.df[column] = self.df[column].apply(
                lambda x: [item.replace(" ", "") for item in x]
            )

        self.df["content"] = (
            self.df["cast"]
            + self.df["crew"]
            + self.df["genres"]
            + self.df["keywords"]
        )
        self.df["content"] = self.df["content"].apply(
            lambda x: " ".join(x).lower()
        )

        self.calculate_weighted_rating()

        self.create_similarity_matrix()

    def calculate_weighted_rating(self):
        """Calculate weighted rating based on IMDB formula"""
        v = self.df["vote_count"]
        R = self.df["vote_average"]
        C = self.df["vote_average"].mean()
        m = self.df["vote_count"].quantile(0.70)

        self.df["weighted_rating"] = ((R * v) + (C * m)) / (v + m)

        scaler = MinMaxScaler()
        normalized_scores = scaler.fit_transform(
            self.df[["weighted_rating", "popularity"]]
        )

        self.df["score"] = (
            normalized_scores[:, 0] * 0.5 + normalized_scores[:, 1] * 0.5
        )

    def create_similarity_matrix(self):
        """Create similarity matrix for content-based filtering"""
        cv = CountVectorizer(max_features=5000, stop_words="english")
        vectors = cv.fit_transform(self.df["content"]).toarray()
        self.similarity_matrix = cosine_similarity(vectors)

    def get_content_based_recommendations(self, movie_title, n=5):
        """Get content-based recommendations for a movie"""
        movie_idx = self.df[
            self.df["title"].str.lower() == movie_title.lower()
        ].index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        similarity_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True
        )
        similar_movies = similarity_scores[1 : n + 1]

        recommendations = []
        for idx, score in similar_movies:
            movie_data = self.df.iloc[idx]
            recommendations.append(
                {
                    "title": movie_data["title"],
                    "similarity_score": round(score * 100, 2),
                    "genres": movie_data["genres"],
                    "vote_average": movie_data["vote_average"],
                }
            )
        return recommendations

    def get_top_rated_movies(self, n=10):
        """Get top rated movies based on weighted score"""
        return self.df.nlargest(n, "score")[
            ["title", "vote_average", "vote_count", "popularity", "score"]
        ]


def main():
    st.set_page_config(page_title="Movie Recommender System", layout="wide")

    st.title("🎬 Movie Recommender System")

    # Initialize recommender
    @st.cache_resource
    def load_recommender():
        return MovieRecommender(
            "dataset/tmdb_5000_movies.csv",
            "dataset/tmdb_5000_credits.csv",
        )

    try:
        recommender = load_recommender()

        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to", ["Movie Recommendations", "Top Rated Movies"]
        )

        if page == "Movie Recommendations":
            st.header("Get Personalized Movie Recommendations")

            movie_list = recommender.df["title"].tolist()
            selected_movie = st.selectbox(
                "Select or type a movie you like:", options=movie_list
            )

            if st.button("Get Recommendations"):
                with st.spinner("Finding similar movies..."):
                    recommendations = (
                        recommender.get_content_based_recommendations(
                            selected_movie
                        )
                    )

                    st.subheader("Here are some movies you might like:")

                    # Display recommendations in a grid
                    cols = st.columns(len(recommendations))
                    for col, movie in zip(cols, recommendations):
                        with col:
                            st.markdown(f"**{movie['title']}**")
                            st.write(
                                f"Similarity: {movie['similarity_score']}%"
                            )
                            st.write(f"Rating: {movie['vote_average']}/10")
                            st.write("Genres: " + ", ".join(movie["genres"]))

        else:
            st.header("Top Rated Movies")

            num_movies = st.slider("Number of movies to display:", 5, 20, 10)

            top_movies = recommender.get_top_rated_movies(num_movies)

            fig = px.bar(
                top_movies,
                x="title",
                y="score",
                title=f"Top {num_movies} Movies by Combined Score",
                labels={"title": "Movie Title", "score": "Combined Score"},
                color="popularity",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Detailed Rankings")
            st.dataframe(
                top_movies.style.format(
                    {
                        "vote_average": "{:.1f}",
                        "popularity": "{:.1f}",
                        "score": "{:.3f}",
                    }
                )
            )

    except FileNotFoundError:
        st.error(
            """
        Error: Dataset files not found. Please ensure you have the following files in your directory:
        - tmdb_5000_movies.csv
        - tmdb_5000_credits.csv
        """
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
