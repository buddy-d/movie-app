import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page title
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)
# Load data
df = pd.read_csv("movies.csv")

# Check columns
if "title" not in df.columns or "description" not in df.columns:
    st.error("CSV must contain 'title' and 'description' columns")
    st.stop()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["description"])

# Similarity calculation
similarity = cosine_similarity(tfidf_matrix)

# Movie selection
movie_name = st.selectbox("Select a movie", df["title"])

# Description
st.markdown("### Movie Description")
st.info(df[df["title"] == movie_name]["description"].values[0])

# Recommendation logic
if st.button("Recommend"):
    movie_index = df[df["title"] == movie_name].index[0]
    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    st.subheader("Recommended Movies:")
    for i in scores:
        st.write(df.iloc[i[0]]["title"])

#Footer
st.markdown("---")
st.markdown("🚀 Built by DK Mini Project")
