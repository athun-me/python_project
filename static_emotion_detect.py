import cv2
from fer import FER
import tensorflow as tf

# Force TensorFlow to use CPU (avoid CUDA errors)
tf.config.set_visible_devices([], 'GPU')

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_emotion(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image.")
        return

    # Convert to grayscale for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using OpenCV's face detector
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected using OpenCV. Trying FER...")
    
    # Initialize FER detector
    detector = FER()
    
    # Detect emotions
    emotions = detector.detect_emotions(image)

    if not emotions:
        print("No face detected or emotion could not be determined.")
        return
    
    # Get the most confident emotion
    for face in emotions:
        emotion, score = max(face["emotions"].items(), key=lambda item: item[1])
        if emotion == "neutral":
            emotion = "calm or neutral"
        print(f"Detected Emotion: {emotion} ({score:.2f})")

# Example usage
image_path = "/home/athun/pythonPP/calm.jpeg"  

detect_emotion(image_path)
