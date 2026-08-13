import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "raw"


def load_data():

    # ----------------------------
    # Read only required columns
    # ----------------------------

    df_basics = pd.read_csv(
        DATA_DIR / "title.basics.tsv" / "title.basics.tsv",
        sep="\t",
        usecols=[
            "tconst",
            "titleType",
            "primaryTitle",
            "originalTitle",
            "genres"
        ],
        low_memory=False
    )

    df_ratings = pd.read_csv(
        DATA_DIR / "title.ratings.tsv" / "title.ratings.tsv",
        sep="\t",
        usecols=[
            "tconst",
            "averageRating",
            "numVotes"
        ]
    )

    df_crew = pd.read_csv(
        DATA_DIR / "title.crew.tsv" / "title.crew.tsv",
        sep="\t",
        usecols=[
            "tconst",
            "directors",
            "writers"
        ]
    )

    df_names = pd.read_csv(
        DATA_DIR / "name.basics.tsv" / "name.basics.tsv",
        sep="\t",
        usecols=[
            "nconst",
            "primaryName"
        ]
    )

    # ----------------------------
    # Filter movies
    # ----------------------------

    df_movies = df_basics[
        df_basics["titleType"] == "movie"
    ].copy()

    df_movies.replace("\\N", np.nan, inplace=True)

    df_movies.dropna(
        subset=["primaryTitle", "originalTitle", "genres"],
        inplace=True
    )

    # ----------------------------
    # Merge ratings
    # ----------------------------

    df_movies = df_movies.merge(
        df_ratings,
        on="tconst",
        how="left"
    )

    # ----------------------------
    # Merge crew
    # ----------------------------

    df_crew.replace("\\N", np.nan, inplace=True)

    df_movies = df_movies.merge(
        df_crew,
        on="tconst",
        how="left"
    )

    # ----------------------------
    # Get director/writer IDs
    # ----------------------------

    director_ids = (
        df_movies["directors"]
        .dropna()
        .str.split(",")
        .explode()
        .unique()
    )

    writer_ids = (
        df_movies["writers"]
        .dropna()
        .str.split(",")
        .explode()
        .unique()
    )

    people_ids = set(director_ids).union(set(writer_ids))

    # Only keep people we actually need
    df_names = df_names[
        df_names["nconst"].isin(people_ids)
    ]

    name_map = (
        df_names
        .set_index("nconst")["primaryName"]
        .to_dict()
    )

    # ----------------------------
    # Map directors and writers
    # ----------------------------

    def map_people(ids):

        if pd.isna(ids):
            return ""

        names = []

        for person_id in str(ids).split(","):
            name = name_map.get(person_id)

            if pd.isna(name):
                names.append(person_id)
            else:
                names.append(str(name))

        return ", ".join(names)

    df_movies["director_names"] = (
        df_movies["directors"].apply(map_people)
    )

    df_movies["writer_names"] = (
        df_movies["writers"].apply(map_people)
    )

    return df_movies


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# Load and prepare data once
df_movies = load_data()

# Combine the information we want the recommender to use
df_movies["features"] = (
    df_movies["genres"].fillna("") + " " +
    df_movies["director_names"].fillna("") + " " +
    df_movies["writer_names"].fillna("")
)
tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    df_movies["features"]
)
movie_indices = pd.Series(
    df_movies.index,
    index=df_movies["primaryTitle"]
).drop_duplicates()
def search_movie(movie_name):
    """
    Search for movies by title.
    """

    matches = df_movies[
        df_movies["primaryTitle"]
        .str.contains(movie_name, case=False, na=False)
    ]

    return matches[
        ["tconst", "primaryTitle", "genres"]
    ].head(10).to_dict(orient="records")


def recommend(movie_name):
    """
    Return top 5 similar movies.
    """

    matches = df_movies[
        df_movies["primaryTitle"].str.lower() == movie_name.lower()
    ]

    if matches.empty:
        return []

    # Use the first exact title match
    idx = matches.index[0]

    similarity_scores = linear_kernel(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[-6:][::-1]

    similar_indices = [
        i for i in similar_indices
        if i != idx
    ][:5]

    return df_movies.iloc[
        similar_indices
    ][
        ["tconst", "primaryTitle", "genres"]
    ].to_dict(orient="records")


if __name__ == "__main__":
    print(search_movie("War"))
    print(recommend("War"))