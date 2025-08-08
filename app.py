import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load pickled data
@st.cache_resource
def load_pickles():
    with open('user_item_matrix.pkl', 'rb') as f:
        user_item_matrix = pickle.load(f)
    with open('user_similarity.pkl', 'rb') as f:
        user_similarity_df = pickle.load(f)
    with open('predicted_svd.pkl', 'rb') as f:
        predicted_df = pickle.load(f)
    with open('movies.pkl', 'rb') as f:
        movies = pickle.load(f)

    # ✅ Ensure predicted_df is always a DataFrame
    if isinstance(predicted_df, np.ndarray):
        predicted_df = pd.DataFrame(
            predicted_df,
            index=user_item_matrix.index,
            columns=user_item_matrix.columns
        )

    return user_item_matrix, user_similarity_df, predicted_df, movies

user_item_matrix, user_similarity_df, predicted_df, movies = load_pickles()

# Recommender functions
def recommend_user_based(user_id, top_n=5):
    sim_scores = user_similarity_df[user_id]
    sim_users = sim_scores.sort_values(ascending=False)[1:]
    weighted_sum = np.dot(sim_users.values, user_item_matrix.loc[sim_users.index])
    sim_sum = np.array([np.abs(sim_users).sum()] * user_item_matrix.shape[1])
    scores = weighted_sum / sim_sum
    user_rated = user_item_matrix.loc[user_id]
    unseen_mask = user_rated == 0
    recommendations = pd.Series(scores, index=user_item_matrix.columns)[unseen_mask]
    top_recommendations = recommendations.sort_values(ascending=False).head(top_n)
    return movies[movies['movie_id'].isin(top_recommendations.index)]

def recommend_svd(user_id, top_n=5):
    user_preds = predicted_df.loc[user_id]
    user_actual = user_item_matrix.loc[user_id]
    unseen = user_actual[user_actual == 0]
    recommendations = user_preds[unseen.index].sort_values(ascending=False).head(top_n)
    return movies[movies['movie_id'].isin(recommendations.index)]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.markdown("Using pre-trained data stored in pickle files.")

user_id = st.selectbox("Select User ID", user_item_matrix.index.tolist())
method = st.radio("Recommendation Method", ["User-Based Collaborative Filtering", "Matrix Factorization (SVD)"])
top_n = st.slider("Number of Recommendations", 1, 20, 5)

if st.button("Recommend Movies"):
    st.write(f"### Recommended Movies for User {user_id}")
    if method == "User-Based Collaborative Filtering":
        recs = recommend_user_based(user_id, top_n)
    else:
        recs = recommend_svd(user_id, top_n)

    st.table(recs[['movie_id', 'title']].reset_index(drop=True))
