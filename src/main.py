import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

sns.set_theme(style="whitegrid")

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# Cleaning
print(ratings.isnull().sum(), movies.isnull().sum())
ratings = ratings.drop_duplicates().dropna()
movies = movies.drop_duplicates().dropna()

# Feature engineering
ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings["year"] = ratings["date"].dt.year
movies["release_year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
movies["genre_list"] = movies["genres"].str.split("|")

# Merge for analysis
df = ratings.merge(movies, on="movieId")
print(df.shape)

# Movie-level stats
movie_stats = df.groupby(["movieId", "title"]).agg(
    avg_rating=("rating", "mean"),
    num_ratings=("rating", "count")
).reset_index()

# Top movies (need enough ratings to be trustworthy)
top_movies = (movie_stats[movie_stats.num_ratings >= 50]
              .sort_values("avg_rating", ascending=False).head(10))
most_rated = movie_stats.sort_values("num_ratings", ascending=False).head(10)

# Genre analysis
genre_df = df.explode("genre_list")
genre_counts = genre_df["genre_list"].value_counts()
genre_avg = genre_df.groupby("genre_list")["rating"].mean().sort_values()

print(top_movies, most_rated, genre_counts, sep="\n\n")

# User-item matrix (rows = users, columns = movies)
user_item = ratings.pivot_table(index="userId", columns="movieId", values="rating")

# Mean-center each user's ratings (removes "harsh vs generous rater" bias)
user_means = user_item.mean(axis=1)
centered = user_item.sub(user_means, axis=0).fillna(0)

# User-user cosine similarity
user_sim = pd.DataFrame(
    cosine_similarity(centered),
    index=user_item.index, columns=user_item.index
)

def recommend(user_id, n=10, k=30, min_neighbors=3):
    """Recommend n movies for a user using their k most similar users."""
    sims = user_sim[user_id].drop(user_id).nlargest(k)
    neighbor_ratings = centered.loc[sims.index]

    # Weighted average of neighbors' centered ratings
    scores = neighbor_ratings.T.dot(sims) / sims.abs().sum()
    scores = scores + user_means[user_id]          # add back user's baseline

    # Remove movies already seen
    seen = user_item.loc[user_id].dropna().index
    scores = scores.drop(seen, errors="ignore")

    # Keep movies rated by enough neighbors (avoids flukes)
    n_rated = user_item.loc[sims.index].notna().sum()
    scores = scores[n_rated[scores.index] >= min_neighbors]

    top = scores.nlargest(n).reset_index()
    top.columns = ["movieId", "predicted_rating"]
    top = top.merge(movies[["movieId", "title", "genres"]], on="movieId")
    top["predicted_rating"] = top["predicted_rating"].clip(0.5, 5).round(2)
    top.insert(0, "userId", user_id)
    return top

sample_users = [1, 50, 150, 300, 500]
all_recs = []

for u in sample_users:
    recs = recommend(u)
    all_recs.append(recs)
    print(f"\n=== Top 10 recommendations for User {u} ===")
    print(recs[["title", "genres", "predicted_rating"]].to_string(index=False))

pd.concat(all_recs).to_csv("outputs/recommendations.csv", index=False)

from sklearn.model_selection import train_test_split

train, test = train_test_split(ratings, test_size=0.2, random_state=42)
train_ui = train.pivot_table(index="userId", columns="movieId", values="rating")
t_means = train_ui.mean(axis=1)
t_cent = train_ui.sub(t_means, axis=0).fillna(0)
t_sim = pd.DataFrame(cosine_similarity(t_cent), index=train_ui.index, columns=train_ui.index)

def predict(u, m, k=30):
    if u not in t_sim.index or m not in t_cent.columns:
        return ratings["rating"].mean()
    sims = t_sim[u].drop(u)
    raters = train_ui[m].dropna().index.intersection(sims.index)
    if len(raters) == 0:
        return t_means[u]
    top = sims[raters].nlargest(k)
    denom = top.abs().sum()
    if denom == 0:
        return t_means[u]
    return np.clip(t_means[u] + (t_cent.loc[top.index, m] * top).sum() / denom, 0.5, 5)

sample = test.sample(2000, random_state=1)
preds = [predict(r.userId, r.movieId) for r in sample.itertuples()]
rmse = np.sqrt(np.mean((sample["rating"].values - np.array(preds)) ** 2))
print("RMSE:", round(rmse, 3))

# 1. Rating distribution
plt.figure(figsize=(7, 4))
sns.countplot(x="rating", data=ratings, color="steelblue")
plt.title("Distribution of Ratings")
plt.savefig("charts/1_rating_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

# 2. Most-rated movies
plt.figure(figsize=(9, 5))
sns.barplot(data=most_rated, y="title", x="num_ratings", color="coral")
plt.title("Top 10 Most Popular Movies (by number of ratings)")
plt.savefig("charts/2_most_popular.png", dpi=150, bbox_inches="tight"); plt.close()

# 3. Highest-rated movies (min 50 ratings)
plt.figure(figsize=(9, 5))
sns.barplot(data=top_movies, y="title", x="avg_rating", color="seagreen")
plt.xlim(3.5, 5); plt.title("Top 10 Highest-Rated Movies (50+ ratings)")
plt.savefig("charts/3_top_rated.png", dpi=150, bbox_inches="tight"); plt.close()

# 4. Genre popularity
plt.figure(figsize=(9, 5))
genre_counts.drop("(no genres listed)", errors="ignore").plot(kind="barh", color="slateblue")
plt.title("Number of Ratings per Genre")
plt.gca().invert_yaxis()
plt.savefig("charts/4_genre_counts.png", dpi=150, bbox_inches="tight"); plt.close()

# 5. Average rating by genre
plt.figure(figsize=(9, 5))
genre_avg.drop("(no genres listed)", errors="ignore").plot(kind="barh", color="goldenrod")
plt.xlim(3, 4.2); plt.title("Average Rating by Genre")
plt.savefig("charts/5_genre_avg.png", dpi=150, bbox_inches="tight"); plt.close()

# 6. Ratings over time (trend)
plt.figure(figsize=(9, 4))
ratings.groupby("year").size().plot(marker="o")
plt.title("Number of Ratings per Year"); plt.ylabel("Ratings")
plt.savefig("charts/6_ratings_trend.png", dpi=150, bbox_inches="tight"); plt.close()

# 7. Movies released per decade
plt.figure(figsize=(9, 4))
(movies["release_year"] // 10 * 10).value_counts().sort_index().plot(kind="bar", color="teal")
plt.title("Movies by Release Decade")
plt.savefig("charts/7_decades.png", dpi=150, bbox_inches="tight"); plt.close()

