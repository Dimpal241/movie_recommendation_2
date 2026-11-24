import os
import json
import pickle
import numpy as np
import pandas as pd

# Graceful import: uses SentenceTransformer if installed, otherwise provides a clean TF-IDF fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SBERT = False

# Starter movie database with real TMDB IDs, overviews, genres, and ratings
# This allows the app to work right out of the box, or you can drop your full tmdb_5000_movies.csv here!
SAMPLE_MOVIES = [
    {
        "id": 19995,
        "title": "Avatar",
        "overview": "In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.",
        "genres": ["Action", "Adventure", "Fantasy", "Science Fiction"],
        "vote_average": 7.2,
        "popularity": 150.4
    },
    {
        "id": 157336,
        "title": "Interstellar",
        "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.",
        "genres": ["Adventure", "Drama", "Science Fiction"],
        "vote_average": 8.6,
        "popularity": 180.2
    },
    {
        "id": 27205,
        "title": "Inception",
        "overview": "Cobb, a skilled thief who steals corporate secrets through the use of dream-sharing technology, is given the inverse task of planting an idea into the mind of a C.E.O.",
        "genres": ["Action", "Science Fiction", "Adventure"],
        "vote_average": 8.3,
        "popularity": 167.5
    },
    {
        "id": 155,
        "title": "The Dark Knight",
        "overview": "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets.",
        "genres": ["Drama", "Action", "Crime", "Thriller"],
        "vote_average": 8.5,
        "popularity": 123.1
    },
    {
        "id": 680,
        "title": "Pulp Fiction",
        "overview": "A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer converge in this sprawling, comedic crime caper.",
        "genres": ["Thriller", "Crime"],
        "vote_average": 8.5,
        "popularity": 121.4
    },
    {
        "id": 13,
        "title": "Forrest Gump",
        "overview": "A man with a low IQ has accomplished great things in his life and been present during significant historic events—in each case, far exceeding what anyone imagined he could do.",
        "genres": ["Comedy", "Drama", "Romance"],
        "vote_average": 8.2,
        "popularity": 138.3
    },
    {
        "id": 299536,
        "title": "Avengers: Infinity War",
        "overview": "As the Avengers and their allies have continued to protect the world from threats too large for any one hero to handle, a new danger has emerged from the cosmic shadows: Thanos.",
        "genres": ["Adventure", "Action", "Science Fiction"],
        "vote_average": 8.3,
        "popularity": 190.8
    },
    {
        "id": 496243,
        "title": "Parasite",
        "overview": "All unemployed, Ki-taek's family takes peculiar interest in the wealthy and glamorous Parks for their livelihood until they get entangled in an unexpected incident.",
        "genres": ["Comedy", "Thriller", "Drama"],
        "vote_average": 8.5,
        "popularity": 115.6
    },
    {
        "id": 278,
        "title": "The Shawshank Redemption",
        "overview": "Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison.",
        "genres": ["Drama", "Crime"],
        "vote_average": 8.7,
        "popularity": 142.9
    },
    {
        "id": 105,
        "title": "Back to the Future",
        "overview": "Eighties teenager Marty McFly is accidentally sent back in time to 1955, inadvertently disrupting his parents' first meeting and putting his very existence at stake.",
        "genres": ["Adventure", "Comedy", "Science Fiction"],
        "vote_average": 8.3,
        "popularity": 98.4
    },
    {
        "id": 429,
        "title": "The Good, the Bad and the Ugly",
        "overview": "While the Civil War rages between the Union and the Confederacy, three men—a quiet loner, a ruthless hit man and a Mexican bandit—comb the American Southwest in search of a strongbox containing $200,000 in stolen gold.",
        "genres": ["Western"],
        "vote_average": 8.2,
        "popularity": 75.3
    },
    {
        "id": 550,
        "title": "Fight Club",
        "overview": "A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy.",
        "genres": ["Drama"],
        "vote_average": 8.4,
        "popularity": 112.0
    },
    {
        "id": 120,
        "title": "The Lord of the Rings: The Fellowship of the Ring",
        "overview": "Young hobbit Frodo Baggins, after inheriting a mysterious ring from his uncle Bilbo, must leave his home behind in order to begin an epic quest to destroy it.",
        "genres": ["Adventure", "Fantasy", "Action"],
        "vote_average": 8.4,
        "popularity": 145.7
    },
    {
        "id": 37724,
        "title": "Skyfall",
        "overview": "When Bond's latest assignment goes gravely wrong, agents around the world are exposed and MI6 is attacked, forcing M to relocate the agency.",
        "genres": ["Action", "Adventure", "Thriller"],
        "vote_average": 7.2,
        "popularity": 89.2
    },
    {
        "id": 24428,
        "title": "The Avengers",
        "overview": "When an unexpected enemy emerges and threatens global safety and security, Nick Fury, director of the international peacekeeping agency known as S.H.I.E.L.D., finds himself in need of a team to pull the world back from the brink of disaster.",
        "genres": ["Action", "Adventure", "Science Fiction"],
        "vote_average": 7.7,
        "popularity": 130.5
    },
    {
        "id": 330459,
        "title": "Rogue One: A Star Wars Story",
        "overview": "A rogue band of resistance fighters unite for a daring mission to steal the Death Star plans and bring new hope to the galaxy.",
        "genres": ["Action", "Adventure", "Science Fiction"],
        "vote_average": 7.5,
        "popularity": 105.0
    },
    {
        "id": 8587,
        "title": "The Lion King",
        "overview": "A young lion prince is cast out of his pride by his cruel uncle, who claims he killed his father. While the uncle rules with an iron paw, the prince grows beyond his savannah home.",
        "genres": ["Family", "Animation", "Drama"],
        "vote_average": 8.2,
        "popularity": 110.1
    },
    {
        "id": 11,
        "title": "Star Wars",
        "overview": "Princess Leia is captured and held hostage by the evil Imperial forces in their effort to take over the galactic Empire. Venturesome Luke Skywalker and dashing captain Han Solo team together with the loveable robot duo R2-D2 and C-3PO to rescue the beautiful princess and restore peace and justice in the Empire.",
        "genres": ["Adventure", "Action", "Science Fiction"],
        "vote_average": 8.2,
        "popularity": 125.0
    },
    {
        "id": 769,
        "title": "GoodFellas",
        "overview": "The true story of Henry Hill, a half-Irish, half-Sicilian Brooklyn kid who is adopted by neighbourhood gangsters at an early age and climbs the ranks of a Mafia family.",
        "genres": ["Drama", "Crime"],
        "vote_average": 8.5,
        "popularity": 95.8
    },
    {
        "id": 603,
        "title": "The Matrix",
        "overview": "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a underground rebellion against the vast computer that now rules the world.",
        "genres": ["Action", "Science Fiction"],
        "vote_average": 8.2,
        "popularity": 140.0
    }
]


def load_dataset():
    """
    Checks if tmdb_5000_movies.csv exists in the directory.
    If it exists, parses the full 5,000 dataset; otherwise uses the starter catalog.
    """
    csv_path = "tmdb_5000_movies.csv"
    if os.path.exists(csv_path):
        print(f"Found {csv_path}. Processing full dataset...")
        df = pd.read_csv(csv_path)

        def parse_genres(genre_str):
            try:
                items = json.loads(genre_str)
                return [i['name'] for i in items]
            except Exception:
                return []

        df['genres'] = df['genres'].apply(parse_genres)
        df['overview'] = df['overview'].fillna('')
        df = df[['id', 'title', 'overview', 'genres', 'vote_average', 'popularity']].dropna(subset=['title'])
        return df
    else:
        print("Using starter movie catalog (20 high-profile movies).")
        print("Tip: Drop tmdb_5000_movies.csv in this folder anytime to train on the full 5000 movies!")
        return pd.DataFrame(SAMPLE_MOVIES)


def prepare_and_embed():
    df = load_dataset()

    # Create enriched text for each movie: genres + overview
    df['genre_text'] = df['genres'].apply(lambda g: " ".join(g) if isinstance(g, list) else str(g))
    df['content_text'] = df['title'] + ". Genres: " + df['genre_text'] + ". Plot: " + df['overview']

    if HAS_SBERT:
        print(f"Encoding {len(df)} movies using sentence-transformers/all-MiniLM-L6-v2...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['content_text'].tolist(), show_progress_bar=True)
    else:
        print("Note: sentence-transformers not found in environment. Generating TF-IDF semantic vectors as fallback.")
        print("Tip: Run 'pip install sentence-transformers' anytime for full neural embeddings.")
        tfidf = TfidfVectorizer(max_features=384, stop_words='english')
        embeddings = tfidf.fit_transform(df['content_text'].tolist()).toarray()
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
    
    # Downcast to float32 to reduce memory footprint by 50%
    embeddings = np.array(embeddings, dtype=np.float32)

    # Save processed dataframe and embeddings array
    with open('movies_data.pkl', 'wb') as f:
        pickle.dump(df, f)

    np.save('embeddings.npy', embeddings)

    print("\nProcessing complete!")
    print(f"- Saved movies_data.pkl ({len(df)} records)")
    print(f"- Saved embeddings.npy (shape: {embeddings.shape}, dtype: {embeddings.dtype})")
    print("\nYou are ready to launch the app using: streamlit run app.py")


if __name__ == "__main__":
    prepare_and_embed()
