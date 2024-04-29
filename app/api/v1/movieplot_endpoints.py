from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.distilbert_models import DistilBERTModel

router = APIRouter()

# Initialize model on CPU explicitly
model = DistilBERTModel('../distilbert_models/pytorch_distilbert_movies.bin',
                        '../distilbert_models/vocab_distilbert_movies.bin', 'cpu')


class Plot(BaseModel):
    movie_plot: str


@router.post("/movieplot_predict/")
async def predict_genre(plot: Plot):
    try:
        predictions, probabilities = model.predict(plot.movie_plot)
        # Assuming genre labels mapping is required
        genres = ['Adventure', 'Action', 'Romance', 'Thriller', 'Drama', 'Horror', 'Science Fiction', 'Comedy']
        predicted_genres = [genres[i] for i, pred in enumerate(predictions) if pred == 1]
        return {"genres": predicted_genres, "probabilities": probabilities.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
