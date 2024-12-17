from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=-1).tolist()[0]
    # scores[0] = negative, scores[1] = positive
    sentiment = "positive" if scores[1] > scores[0] else "negative"
    confidence = max(scores)
    return sentiment, confidence
