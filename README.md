# 🎬 Movie Recommendation System  

A **simple yet powerful movie recommender system** built using the **MovieLens 100K dataset**.  
This project implements **two recommendation approaches** – **User-Based Collaborative Filtering** and **Matrix Factorization (SVD)** – and serves them via an **interactive Streamlit web app**.  
Pre-trained models and data are stored using **Pickle** for faster load times.  

---

## 📌 Features  
🔹 **User-Based Collaborative Filtering** – Recommends movies based on similar users' preferences  
🔹 **Matrix Factorization (SVD)** – Leverages latent factors for improved personalization  
🔹 **Pickle-Based Data Storage** – Fast load times using precomputed similarity and prediction matrices  
🔹 **Dynamic Recommendation Control** – Select the number of recommendations  
🔹 **Interactive Streamlit App** – Simple UI to explore recommendations  

---

## 🛠️ Tech Stack  

**Languages & Libraries**  
- Python  
- Pandas, NumPy  
- Scikit-learn  
- SciPy (SVD)  
- Streamlit  
- Pickle  

---

## 📊 Dataset
We used the MovieLens 100K dataset:

Data source: [MovieLens Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)

---

## 📈 Recommendation Approaches
- User-Based Collaborative Filtering
Finds similar users using cosine similarity

Suggests movies liked by similar users

- Matrix Factorization (SVD)
Reduces dimensionality to discover latent features

Predicts ratings for unseen movies

---

## 📷 App Preview
### User selects:

User ID

Recommendation method (User-Based / SVD)

Number of recommendations

### App returns:
A table of recommended movies with IDs and titles.

---

## 🏆 What I Learned
- Building & comparing multiple recommendation algorithms

- Implementing evaluation metrics like Precision@K

- Optimizing performance using pre-trained pickle files

- Deploying ML models via Streamlit

---

## 🚀 How to Run the Project  

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kaifi1199/User-SVD-Based-Movie-Recommendation-System.git
   cd movie-recommendation-system

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py

---
