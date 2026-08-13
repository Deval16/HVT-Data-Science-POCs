from fastapi import FastAPI
from src.recommender import search_movie, recommend


app = FastAPI(
    title="IMDb Movie Recommendation API",
    description="API for searching movies and generating recommendations",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "IMDb Movie Recommendation API is running"
    }


@app.get("/search")
def search(movie_name: str):

    results = search_movie(movie_name)

    return {
        "query": movie_name,
        "results": results
    }


@app.get("/recommend")
def get_recommendations(movie_name: str):

    results = recommend(movie_name)

    return {
        "movie": movie_name,
        "recommendations": results
    }