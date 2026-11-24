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

# Load environment variables from .env file if present
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Configure Streamlit page layout
st.set_page_config(
    page_title="CineMatch AI - Smart Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom clean CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7B2CBF;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 25px;
    }
    .movie-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 15px;
        border: 1px solid #e9ecef;
    }
    .badge {
        background-color: #7B2CBF;
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .score-badge {
        background-color: #2a9d8f;
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load SBERT model or fallback TFIDF vectorizer"""
    if HAS_SBERT:
        return SentenceTransformer('all-MiniLM-L6-v2')
    elif os.path.exists('vectorizer.pkl'):
        with open('vectorizer.pkl', 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_data():
    """Load preprocessed movie catalog and embeddings"""
    if not os.path.exists('movies_data.pkl') or not os.path.exists('embeddings.npy'):
        # Auto-run prepare_data if files are missing
        import prepare_data
        prepare_data.prepare_and_embed()

    with open('movies_data.pkl', 'rb') as f:
        movies_df = pickle.load(f)

    embeddings = np.load('embeddings.npy')
    return movies_df, embeddings


# Fetch poster from TMDB API with graceful fallback
def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        # High quality placeholder if no API key provided
        return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80"

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
    except Exception:
        pass

    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80"


# Generate AI explanation (Gemini API with smart fallback)
def generate_explanation(source_context, target_movie_title, target_overview, target_genres):
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = (
                f"A movie fan is interested in: '{source_context}'.\n"
                f"Explain in exactly ONE friendly, engaging sentence why they would enjoy watching '{target_movie_title}' "
                f"given its plot: '{target_overview}' and genres: {target_genres}. Avoid generic praise."
            )
            res = model.generate_content(prompt)
            if res and res.text:
                return res.text.strip()
        except Exception:
            pass

    # Fallback explanation if Gemini is not configured
    genre_str = ", ".join(target_genres[:2]) if isinstance(target_genres, list) else str(target_genres)
    return f"Recommended for its strong narrative in the {genre_str} space, echoing themes from your query."


# Hybrid Re-ranking algorithm
def compute_hybrid_scores(similarity_vector, movies_df, w_sim=0.70, w_vote=0.20, w_pop=0.10):
    # Min-max normalization for vote_average and popularity
    votes = movies_df['vote_average'].values.astype(np.float32)
    pops = movies_df['popularity'].values.astype(np.float32)

    norm_votes = (votes - votes.min()) / (votes.max() - votes.min() + 1e-6)
    norm_pops = (pops - pops.min()) / (pops.max() - pops.min() + 1e-6)

    # Hybrid multi-factor score
    hybrid_scores = (w_sim * similarity_vector) + (w_vote * norm_votes) + (w_pop * norm_pops)
    return hybrid_scores


# Main Application
def main():
    st.markdown('<div class="main-header">🎬 CineMatch AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Semantic Vector Search & Hybrid Recommendation Engine</div>', unsafe_allow_html=True)

    # Load resources
    with st.spinner("Loading neural embeddings and movie database..."):
        movies_df, embeddings = load_data()
        model = load_model()

    # Sidebar: Control Panel
    st.sidebar.header("⚙️ Recommendation Settings")

    search_mode = st.sidebar.radio(
        "Choose Search Mode:",
        ["🎯 By Similar Movie", "💬 Natural Language Query"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Hybrid Scoring Weights")
    w_sim = st.sidebar.slider("Semantic Similarity Weight", 0.0, 1.0, 0.70, 0.05)
    w_vote = st.sidebar.slider("Audience Rating Weight", 0.0, 1.0, 0.20, 0.05)
    w_pop = st.sidebar.slider("Popularity Weight", 0.0, 1.0, 0.10, 0.05)

    # Normalize weights so they sum to 1
    total_w = w_sim + w_vote + w_pop
    if total_w > 0:
        w_sim, w_vote, w_pop = w_sim / total_w, w_vote / total_w, w_pop / total_w

    top_k = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

    # Run recommendation based on selected mode
    recommended_indices = []
    source_context_desc = ""

    if search_mode == "🎯 By Similar Movie":
        selected_movie = st.selectbox("Select a movie you love:", movies_df['title'].values)

        if st.button("🚀 Find Recommendations", type="primary"):
            movie_idx = movies_df[movies_df['title'] == selected_movie].index[0]
            movie_vector = embeddings[movie_idx].reshape(1, -1)

            # Compute pairwise cosine similarity
            sim_scores = cosine_similarity(movie_vector, embeddings)[0]

            # Calculate hybrid re-ranked scores
            final_scores = compute_hybrid_scores(sim_scores, movies_df, w_sim, w_vote, w_pop)

            # Sort indices (ignoring the query movie itself)
            sorted_indices = np.argsort(final_scores)[::-1]
            recommended_indices = [i for i in sorted_indices if i != movie_idx][:top_k]
            source_context_desc = f"movies like '{selected_movie}'"

    else:
        user_query = st.text_input(
            "Describe the kind of movie you want to watch:",
            placeholder="e.g., A mind-bending sci-fi with space travel and emotional time dilation"
        )

        if st.button("🔍 Search by Plot & Mood", type="primary") and user_query.strip():
            # Embed user natural language query
            if HAS_SBERT and hasattr(model, 'encode'):
                query_vector = model.encode([user_query])
            elif hasattr(model, 'transform'):
                query_vector = model.transform([user_query]).toarray()
            else:
                query_vector = np.zeros((1, embeddings.shape[1]))

            query_vector = np.array(query_vector, dtype=np.float32)

            # Compute semantic similarities
            sim_scores = cosine_similarity(query_vector, embeddings)[0]

            # Apply hybrid ranking
            final_scores = compute_hybrid_scores(sim_scores, movies_df, w_sim, w_vote, w_pop)

            sorted_indices = np.argsort(final_scores)[::-1]
            recommended_indices = sorted_indices[:top_k]
            source_context_desc = f"'{user_query}'"

    # Display Results Grid
    if len(recommended_indices) > 0:
        st.markdown(f"### ✨ Top Recommendations for {source_context_desc}")
        cols = st.columns(top_k)

        for col, idx in zip(cols, recommended_indices):
            movie = movies_df.iloc[idx]
            poster_url = fetch_poster(movie['id'])
            explanation = generate_explanation(
                source_context_desc,
                movie['title'],
                movie['overview'],
                movie['genres']
            )

            with col:
                st.image(poster_url, use_container_width=True)
                st.markdown(f"**{movie['title']}**")
                st.markdown(f"<span class='badge'>⭐ {movie['vote_average']}/10</span>", unsafe_allow_html=True)

                with st.expander("📖 Plot & AI Insight"):
                    st.write(movie['overview'])
                    st.markdown("---")
                    st.markdown(f"💡 *{explanation}*")


if __name__ == "__main__":
    main()
