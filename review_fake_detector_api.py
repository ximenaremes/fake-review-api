# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["deceptive", "truthful"]

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

app = FastAPI()

class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
def predict_review(review: ReviewInput):
    try:
        result = classifier(review.text)[0]
        label = LABELS[int(result['label'].split('_')[-1])] if 'label_' in result['label'] else (
            0 if result['label'].lower() == 'deceptive' else 1
        )
        return {"label": label, "score": float(result['score'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
