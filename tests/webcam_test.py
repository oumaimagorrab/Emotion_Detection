import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    print(f"Erreur : fichier modèle introuvable -> {MODEL_PATH}")
    exit()

print("Chargement du modèle...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Modèle chargé avec succès.")

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Erreur : impossible de charger le détecteur de visage.")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'accéder à la caméra.")
    print("Essaie de remplacer VideoCapture(0) par VideoCapture(1).")
    exit()

print("Caméra démarrée. Appuie sur Q pour quitter.")

prediction_buffer = []
buffer_size = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : lecture caméra impossible.")
        break

    display_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_resized = cv2.resize(face, (260, 260))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # IMPORTANT : pas de division par 255 ici
        face_array = face_rgb.astype("float32")
        face_array = np.expand_dims(face_array, axis=0)

        preds = model.predict(face_array, verbose=0)[0]

        prediction_buffer.append(preds)
        if len(prediction_buffer) > buffer_size:
            prediction_buffer.pop(0)

        avg_preds = np.mean(prediction_buffer, axis=0)

        pred_index = np.argmax(avg_preds)
        pred_label = class_names[pred_index]
        confidence = avg_preds[pred_index] * 100

        top3_idx = np.argsort(avg_preds)[-3:][::-1]

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{pred_label} ({confidence:.1f}%)"
        cv2.putText(
            display_frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        y_text = y2 + 25
        for i in top3_idx:
            line = f"{class_names[i]}: {avg_preds[i]*100:.1f}%"
            cv2.putText(
                display_frame,
                line,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            y_text += 22

    cv2.imshow("Emotion Detection - Webcam", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()