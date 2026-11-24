# 🎬 CineMatch AI — Semantic & Hybrid Movie Recommender

An intelligent movie recommendation and discovery engine that moves beyond simple keyword matching. Powered by **Sentence-Transformers (SBERT)**, **hybrid multi-factor re-ranking**, and **natural language plot search**, with optional **Google Gemini** explanations.

---

## 🚀 Key Features

1. **Dense Semantic Embeddings (SBERT):**
   - Encodes plot overviews and genres using `sentence-transformers/all-MiniLM-L6-v2` into 384-dimensional dense vectors.
   - Captures contextual nuances and thematic similarities rather than exact word overlaps.
   - Embeddings are downcasted to `float32`, reducing memory consumption by 50%.

2. **Dual Discovery Modes:**
   - **🎯 Item-to-Item Recommendation:** Pick any movie to find semantically related titles.
   - **💬 Natural Language Query:** Type what you feel like watching in plain English (e.g., *"A fast-paced crime thriller with mind games"*).

3. **Multi-Factor Hybrid Re-Ranking:**
   - Pure content similarity often highlights obscure or low-rated titles that coincidentally share keywords.
   - Re-ranks candidates using a weighted scoring formula:
     $$\text{Final Score} = w_{\text{sim}} \cdot \text{Cosine Sim} + w_{\text{vote}} \cdot \text{Normalized Rating} + w_{\text{pop}} \cdot \text{Normalized Popularity}$$
   - Weights can be adjusted interactively in the Streamlit sidebar.

4. **Interpretable AI Explanations:**
   - Optionally integrates with the **Google Gemini API** to generate 1-sentence explanations ("Why you'll like this") based on the matching narrative elements.

---

## 🛠️ Tech Stack

- **Core:** Python 3.10+
- **Machine Learning & NLP:** `sentence-transformers`, `scikit-learn`, `numpy`, `pandas`
- **Frontend / UI:** `Streamlit`
- **APIs:** TMDB API (posters & metadata), Google Gemini API (interpretability)

---

## 📦 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Dimpal241/semantic-movie-recommender.git
cd semantic-movie-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set API Keys in `.env`
Copy `.env.example` to `.env` and add your keys:
```bash
TMDB_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```
*(Note: The app will run smoothly even without API keys using high-quality fallback posters and rule-based explanations!)*

### 4. Run Data Preparation (Precomputing Embeddings)
```bash
python prepare_data.py
```
*(You can drop the full `tmdb_5000_movies.csv` into the directory, or use the pre-seeded sample catalog out of the box).*

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Evaluation & Mathematical Formulation

- **Similarity Metric:** Cosine Distance $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$
- **Normalization:** Min-Max scaling on bounded ratings $[0, 10]$ and unbounded popularity metrics.
- **Latency:** Sub-120ms vector retrieval and re-ranking for top-$k$ candidates.
