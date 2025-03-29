import cv2
from fer import FER

detector = FER()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotions = detector.detect_emotions(frame)

    for face in emotions:
        (x, y, w, h) = face["box"]
        
        sorted_emotions = sorted(face["emotions"].items(), key=lambda item: item[1], reverse=True)
        top_emotion, top_score = sorted_emotions[0]
        second_emotion, second_score = sorted_emotions[1] if len(sorted_emotions) > 1 else ("", 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - 40), (x + w, y), (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"{top_emotion.capitalize()} ({top_score:.2f})", (x, y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if second_score > 0:
            cv2.putText(frame, f"{second_emotion.capitalize()} ({second_score:.2f})", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
