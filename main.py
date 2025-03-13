import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df = pd.read_csv("data/books.csv")

# Drop rows with missing values
df = df.dropna()

# Drop the 'price' column
df = df.drop(columns=["price"])

# Ensure ratings are valid (non-NaN and numeric)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"])

# Extract genres and create a binary matrix
df["genres"] = df["genres"].apply(lambda x: x.split(","))
mlb_genres = MultiLabelBinarizer()
genre_matrix = pd.DataFrame(mlb_genres.fit_transform(df["genres"]), columns=mlb_genres.classes_)

# Extract authors and create a binary matrix
mlb_authors = MultiLabelBinarizer()
author_matrix = pd.DataFrame(mlb_authors.fit_transform(df[["author"]]), columns=mlb_authors.classes_)

# Normalize ratings
scaler = MinMaxScaler()
df["normalized_rating"] = scaler.fit_transform(df[["rating"]])

# Combine features into a single matrix
feature_matrix = pd.concat([genre_matrix, author_matrix, df["normalized_rating"]], axis=1)

# Fill NaN values with 0
feature_matrix = feature_matrix.fillna(0)

# Collaborative Filtering: Use Nearest Neighbors on user ratings
collab_model = NearestNeighbors(metric="cosine", algorithm="brute")
collab_model.fit(df[["rating"]])
_, collab_indices = collab_model.kneighbors(df[["rating"]], n_neighbors=5)

# Content-Based Filtering: Use Nearest Neighbors on book features
content_model = NearestNeighbors(metric="cosine", algorithm="brute")
content_model.fit(feature_matrix)
_, content_indices = content_model.kneighbors(feature_matrix, n_neighbors=5)

# Hybrid Model: Combine top recommendations from both approaches
hybrid_recommendations = []
for i in range(len(df)):
    try:
        collab_recs = df.iloc[collab_indices[i]]["title"].tolist()
        content_recs = df.iloc[content_indices[i]]["title"].tolist()
        hybrid_recs = list(set(collab_recs + content_recs))[:5]  # Combine and deduplicate
        hybrid_recommendations.append(hybrid_recs)
    except IndexError:
        # Skip invalid indices
        continue

# Display hybrid recommendations for the first few books
for i, recs in enumerate(hybrid_recommendations[:5]):
    print(f"Book: {df.iloc[i]['title']}")
    print(f"Recommendations: {recs}")
    print()