
import os
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "outputs" / "sentiment-model"))
app = FastAPI(title="DistilBERT sentiment API")
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


@app.on_event("startup")
def load_model_once():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()


@app.post("/predict")
def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    index = int(probabilities.argmax())
    labels = model.config.id2label
    return {"label": labels[index], "confidence": float(probabilities[index])}
