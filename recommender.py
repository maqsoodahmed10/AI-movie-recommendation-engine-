import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.content_sim_matrix = None
        self.collab_sim_matrix = None
        self.user_item_matrix = None

    def load_and_preprocess_data(self, movies_path="movies.csv", ratings_path="ratings.csv"):
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)

        # Replace pipe '|' in genres with spaces for TF-IDF processing
        self.movies_df['genres_cleaned'] = self.movies_df['genres'].str.replace('|', ' ', regex=False).fillna('')

    def build_content_based_model(self):
        """Content Filtering: Recommend based on Genre similarity"""
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['genres_cleaned'])
        self.content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def build_collaborative_model(self, n_components=20):
        """Collaborative Filtering: Recommend based on user rating patterns"""
        self.user_item_matrix = self.ratings_df.pivot(
            index='movieId', columns='userId', values='rating'
        ).fillna(0)

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        latent_matrix = svd.fit_transform(self.user_item_matrix)
        self.collab_sim_matrix = cosine_similarity(latent_matrix)

    def recommend_hybrid(self, movie_title, top_n=5, alpha=0.5):
        if movie_title not in self.movies_df['title'].values:
            return pd.DataFrame()

        idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        movie_id = self.movies_df.loc[idx, 'movieId']

        content_scores = self.content_sim_matrix[idx]

        if movie_id in self.user_item_matrix.index:
            collab_idx = self.user_item_matrix.index.get_loc(movie_id)
            collab_scores_raw = self.collab_sim_matrix[collab_idx]
            
            collab_scores = np.zeros(len(self.movies_df))
            for c_i, m_id in enumerate(self.user_item_matrix.index):
                m_df_idx = self.movies_df[self.movies_df['movieId'] == m_id].index
                if not m_df_idx.empty:
                    collab_scores[m_df_idx[0]] = collab_scores_raw[c_i]
        else:
            collab_scores = content_scores

        # Combine weighted similarity scores
        combined_scores = (alpha * content_scores) + ((1 - alpha) * collab_scores)

        sim_indices = np.argsort(combined_scores)[::-1]
        sim_indices = [i for i in sim_indices if i != idx][:top_n]

        results = self.movies_df.iloc[sim_indices][['movieId', 'title', 'genres']].copy()
        results['similarity_score'] = combined_scores[sim_indices]
        return results

    def save_artifacts(self, file_path="recommender_model.pkl"):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_artifacts(file_path="recommender_model.pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)