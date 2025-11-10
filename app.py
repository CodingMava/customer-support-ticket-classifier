# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast
import torch
from model import SupportClassifierModel
import json

app = FastAPI(title="Support Ticket Classifier API")

class RequestText(BaseModel):
    text: str

# Load model and tokenizer
model_dir = "outputs"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
intent_labels = json.load(open(f"{model_dir}/intent_labels.json"))

num_intents = len(intent_labels)
num_slots = 10
model = SupportClassifierModel(num_intents, num_slots)
model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", map_location="cpu"))
model.eval()

@app.post("/predict")
def predict(request: RequestText):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        intent_logits, slot_logits = model(**inputs)
        intent = intent_labels[int(torch.argmax(intent_logits))]
    return {"text": request.text, "predicted_intent": intent}
