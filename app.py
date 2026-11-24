import os
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

# Load environment variables
load_dotenv()

# Verified working TMDB API key by default with env override
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "25f14bdb1ae3439e6c84a5c154457e32")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Page Configuration
st.set_page_config(
    page_title="CineMatch AI — Smart Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Clean, professional styling (Netflix / Letterboxd aesthetic)
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #8A2BE2;
        margin-bottom: 2px;
        text-align: center;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 30px;
    }
    .poster-img {
        border-radius: 10px;
        transition: transform 0.2s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .movie-title {
        font-size: 1rem;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
        line-height: 1.3;
        height: 2.6em;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .rating-badge {
        background-color: #f59e0b;
        color: #111827;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 6px;
    }
    .genre-text {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 6px;
    }
    .insight-box {
        font-size: 0.82rem;
        color: #d1d5db;
        background-color: rgba(138, 43, 226, 0.1);
        border-left: 3px solid #8A2BE2;
        padding: 6px 8px;
        border-radius: 4px;
        margin-top: 6px;
        line-height: 1.35;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load neural embedding model or fallback TF-IDF"""
    if HAS_SBERT:
        return SentenceTransformer('all-MiniLM-L6-v2')
    elif os.path.exists('vectorizer.pkl'):
        with open('vectorizer.pkl', 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_data():
    """Load movie catalog and precomputed embeddings"""
    need_prepare = not os.path.exists('movies_data.pkl') or not os.path.exists('embeddings.npy')

    if not need_prepare:
        try:
            emb = np.load('embeddings.npy')
            # If SBERT is available, embeddings must have 384 dimensions
            if HAS_SBERT and emb.shape[1] != 384:
                need_prepare = True
            elif not HAS_SBERT and os.path.exists('vectorizer.pkl'):
                with open('vectorizer.pkl', 'rb') as f:
                    vec = pickle.load(f)
                    if hasattr(vec, 'vocabulary_') and emb.shape[1] != len(vec.vocabulary_):
                        need_prepare = True
        except Exception:
            need_prepare = True

    if need_prepare:
        import prepare_data
        prepare_data.prepare_and_embed()

    with open('movies_data.pkl', 'rb') as f:
        movies_df = pickle.load(f)

    embeddings = np.load('embeddings.npy')
    return movies_df, embeddings


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    """Fetch high-res poster from TMDB API with reliable fallback"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    # High-quality cinema fallback if network fails
    return "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80"


def generate_explanation(source_context, target_movie_title, target_overview, target_genres):
    """Generate concise 1-sentence recommendation reason"""
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = (
                f"A user liked '{source_context}'. Explain in exactly 1 brief sentence (under 18 words) "
                f"why they will enjoy '{target_movie_title}' based on plot: {target_overview}."
            )
            res = model.generate_content(prompt)
            if res and res.text:
                return res.text.strip()
        except Exception:
            pass

    # Clean, human-like fallback
    genres_str = " • ".join(target_genres[:2]) if isinstance(target_genres, list) else str(target_genres)
    return f"Shares compelling {genres_str} storytelling with strong audience acclaim."


def compute_hybrid_scores(similarity_vector, movies_df, w_sim=0.65, w_vote=0.25, w_pop=0.10):
    """Multi-factor hybrid re-ranking: Similarity + Rating + Popularity"""
    votes = movies_df['vote_average'].values.astype(np.float32)
    pops = movies_df['popularity'].values.astype(np.float32)

    norm_votes = (votes - votes.min()) / (votes.max() - votes.min() + 1e-6)
    norm_pops = (pops - pops.min()) / (pops.max() - pops.min() + 1e-6)

    return (w_sim * similarity_vector) + (w_vote * norm_votes) + (w_pop * norm_pops)


def main():
    # Hero Title
    st.markdown('<div class="main-title">🎬 CineMatch AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Semantic Plot Search & Hybrid Recommendation Engine</div>', unsafe_allow_html=True)

    # Load data and model
    movies_df, embeddings = load_data()
    model = load_model()

    # Sidebar: Simple, human-friendly settings
    st.sidebar.markdown("### ⚙️ Recommendations")

    pref_mode = st.sidebar.selectbox(
        "Recommendation Style:",
        [
            "✨ Balanced (High Match & High Rating)",
            "⭐ Top Rated Masterpieces (8.0+ Focus)",
            "🔍 Pure Plot Similarity"
        ],
        index=0
    )

    if "Balanced" in pref_mode:
        w_sim, w_vote, w_pop = 0.65, 0.25, 0.10
    elif "Top Rated" in pref_mode:
        w_sim, w_vote, w_pop = 0.40, 0.50, 0.10
    else:
        w_sim, w_vote, w_pop = 0.90, 0.05, 0.05

    num_recommendations = st.sidebar.slider("Number of Recommendations", 4, 12, 8, step=4)

    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by SBERT Neural Embeddings & TMDB 5000.")

    # Main Area: Clean, centered search tabs
    tab_movie, tab_query = st.tabs(["🎯 Find by Favorite Movie", "💬 Search by Plot & Mood"])

    recommended_indices = []
    source_context_desc = ""

    # Tab 1: Movie to Movie
    with tab_movie:
        col_select, col_btn = st.columns([4, 1])
        with col_select:
            selected_movie = st.selectbox(
                "Choose a movie you enjoyed:",
                movies_df['title'].values,
                index=0,
                label_visibility="collapsed"
            )
        with col_btn:
            btn_movie = st.button("Recommend", type="primary", use_container_width=True)

        if btn_movie:
            movie_idx = movies_df[movies_df['title'] == selected_movie].index[0]
            movie_vector = embeddings[movie_idx].reshape(1, -1)

            sim_scores = cosine_similarity(movie_vector, embeddings)[0]
            final_scores = compute_hybrid_scores(sim_scores, movies_df, w_sim, w_vote, w_pop)

            sorted_indices = np.argsort(final_scores)[::-1]
            recommended_indices = [i for i in sorted_indices if i != movie_idx][:num_recommendations]
            source_context_desc = selected_movie

    # Tab 2: Natural Language Plot Search
    with tab_query:
        col_input, col_qbtn = st.columns([4, 1])
        with col_input:
            user_query = st.text_input(
                "Describe what kind of story you want to watch:",
                placeholder="e.g. A mind-bending sci-fi space journey with emotional time dilation",
                label_visibility="collapsed"
            )
        with col_qbtn:
            btn_query = st.button("Discover", type="primary", use_container_width=True)

        if btn_query and user_query.strip():
            # If model supports SBERT and embeddings are 384-d
            if HAS_SBERT and hasattr(model, 'encode') and embeddings.shape[1] == 384:
                query_vector = model.encode([user_query])
            elif hasattr(model, 'transform'):
                query_vector = model.transform([user_query]).toarray()
            elif os.path.exists('vectorizer.pkl'):
                with open('vectorizer.pkl', 'rb') as f:
                    vec = pickle.load(f)
                    query_vector = vec.transform([user_query]).toarray()
            elif HAS_SBERT and hasattr(model, 'encode'):
                # In case embeddings are not 384-d yet, re-embed movies catalog to match
                import prepare_data
                prepare_data.prepare_and_embed()
                movies_df, embeddings = load_data()
                query_vector = model.encode([user_query])
            else:
                query_vector = np.zeros((1, embeddings.shape[1]))

            query_vector = np.array(query_vector, dtype=np.float32)

            # Safeguard against any remaining shape discrepancy
            if query_vector.shape[1] != embeddings.shape[1]:
                import prepare_data
                prepare_data.prepare_and_embed()
                st.cache_data.clear()
                st.rerun()

            sim_scores = cosine_similarity(query_vector, embeddings)[0]
            final_scores = compute_hybrid_scores(sim_scores, movies_df, w_sim, w_vote, w_pop)

            sorted_indices = np.argsort(final_scores)[::-1]
            recommended_indices = sorted_indices[:num_recommendations]
            source_context_desc = f'"{user_query}"'

    # Results Display: Clean 4-Cards per Row Grid (Never Squished!)
    if len(recommended_indices) > 0:
        st.markdown(f"#### 🍿 Recommended Movies based on {source_context_desc}")
        st.markdown("<br>", unsafe_allow_html=True)

        cards_per_row = 4
        for row_start in range(0, len(recommended_indices), cards_per_row):
            row_items = recommended_indices[row_start : row_start + cards_per_row]
            cols = st.columns(cards_per_row)

            for col, idx in zip(cols, row_items):
                movie = movies_df.iloc[idx]
                poster_url = fetch_poster(movie['id'])
                genres = " • ".join(movie['genres'][:2]) if isinstance(movie['genres'], list) else str(movie['genres'])
                insight = generate_explanation(source_context_desc, movie['title'], movie['overview'], movie['genres'])

                with col:
                    # Poster
                    st.image(poster_url, use_container_width=True)
                    # Title
                    st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
                    # Rating Badge & Genres
                    st.markdown(f"<span class='rating-badge'>★ {movie['vote_average']:.1f}</span> <span class='genre-text'>• {genres}</span>", unsafe_allow_html=True)
                    # Clean Highlight Insight
                    st.markdown(f"<div class='insight-box'>💡 {insight}</div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
