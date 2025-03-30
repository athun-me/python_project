from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from fer import FER

app = Flask(__name__)

def detect_emotion(image_buffer):

    file_bytes = np.frombuffer(image_buffer.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image format"}
    
    detector = FER()
    emotions = detector.detect_emotions(image)
    
    if not emotions:
        return {"error": "No face detected"}
    
    face = emotions[0]
    emotion, score = max(face["emotions"].items(), key=lambda item: item[1])
    
    if emotion == "neutral":
        emotion = "calm or neutral"
    
    return {"emotion": emotion, "score": round(score, 2)}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    result = detect_emotion(file)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
