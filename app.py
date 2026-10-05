import streamlit as st
import pandas as pd
from recommender import MovieRecommender

# Set up page config
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS INJECTION ---
st.markdown("""
    <style>
    /* Dark Theme & Modern Styling */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Movie Card Styling */
    .movie-card {
        background-color: #1e222d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 15px;
        border: 1px solid #2e3440;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        border-color: #e50914; /* Netflix Red Accent */
    }
    
    .rank-badge {
        background-color: #e50914;
        color: white;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .match-score {
        color: #46d369; /* Streaming Green Match Score */
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 6px;
    }
    
    .genre-chip {
        background-color: #2b303c;
        color: #a0aab8;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        display: inline-block;
        margin-right: 4px;
        margin-top: 4px;
    }
    
    .movie-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
        min-height: 48px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ENGINE ---
@st.cache_resource
def get_engine():
    try:
        return MovieRecommender.load_artifacts("recommender_model.pkl")
    except Exception:
        engine = MovieRecommender()
        engine.load_and_preprocess_data("movies.csv", "ratings.csv")
        engine.build_content_based_model()
        engine.build_collaborative_model()
        engine.save_artifacts("recommender_model.pkl")
        return engine

with st.spinner("Initializing CineMatch Engine..."):
    engine = get_engine()

movie_list = engine.movies_df['title'].values

# --- SIDEBAR ---
st.sidebar.title("🎬 CineMatch AI")
st.sidebar.caption("Personalized ML Recommendation System")
st.sidebar.markdown("---")

selected_movie = st.sidebar.selectbox("🎯 Choose a movie you love:", movie_list)
top_k = st.sidebar.slider("🍿 Number of recommendations:", 3, 10, 5)

strategy = st.sidebar.radio(
    "🤖 Algorithm Preference:",
    ["Hybrid (Balanced)", "Content-Based (Genres)", "Collaborative (User Ratings)"]
)

alpha_map = {
    "Hybrid (Balanced)": 0.5,
    "Content-Based (Genres)": 1.0,
    "Collaborative (User Ratings)": 0.0
}

# --- MAIN PAGE HERO HEADER ---
st.title("🍿 AI Movie Recommendation Engine")
st.write("Leveraging Machine Learning to analyze genre features and user rating similarity patterns.")
st.markdown("---")

# --- RECOMMENDATION LOGIC & UI DISPLAY ---
if selected_movie:
    st.subheader(f"Recommendations based on **{selected_movie}**")
    
    recs = engine.recommend_hybrid(selected_movie, top_n=top_k, alpha=alpha_map[strategy])
    
    if recs.empty:
        st.warning("No recommendations found.")
    else:
        # Display cards in a multi-column grid
        cols = st.columns(min(len(recs), 5))
        
        for idx, (_, row) in enumerate(recs.iterrows()):
            col_index = idx % 5
            
            # Format genres into badge chips
            genres_split = row['genres'].split('|') if isinstance(row['genres'], str) else []
            genre_chips_html = "".join([f'<span class="genre-chip">{g}</span>' for g in genres_split])
            
            with cols[col_index]:
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="rank-badge">Rank #{idx+1}</div>
                        <div class="movie-title">{row['title']}</div>
                        <div class="match-score">🎯 {row['similarity_score']:.1%} Match</div>
                        <div>{genre_chips_html}</div>
                    </div>
                """, unsafe_allow_html=True)