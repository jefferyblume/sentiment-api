from flask import Flask, request, jsonify
from model.load_model import load_sentiment_model, predict_sentiment

app = Flask(__name__)

# Load model and tokenizer once at startup
tokenizer, model = load_sentiment_model()

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    
    text = data["text"]
    sentiment, confidence = predict_sentiment(text, tokenizer, model)
    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
