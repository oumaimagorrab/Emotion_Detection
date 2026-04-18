from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
import base64
from io import BytesIO
from PIL import Image
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../frontend/templates",
            static_folder="../frontend/static")

# Configuration
MODEL_PATH = "model.keras"
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Chargement du modèle au démarrage
logger.info("Chargement du modèle...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
logger.info("✅ Modèle chargé avec succès")

# Chargement du détecteur de visage
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Buffer pour lisser les prédictions
prediction_buffer = {}
buffer_size = 5

def decode_base64_image(base64_string):
    """Décode une image base64 en array numpy"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        image_data = data['image']
        frame = decode_base64_image(image_data)
        
        # Détection des visages
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        results = []
        
        for idx, (x, y, w, h) in enumerate(faces):
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face = frame[y1:y2, x1:x2]
            
            if face.size == 0:
                continue
            
            # Prétraitement
            face_resized = cv2.resize(face, (260, 260))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_array = face_rgb.astype("float32")
            face_array = np.expand_dims(face_array, axis=0)
            
            # Prédiction
            preds = model.predict(face_array, verbose=0)[0]
            
            # Buffer pour chaque visage (utiliser l'index comme ID simple)
            face_id = f"face_{idx}"
            if face_id not in prediction_buffer:
                prediction_buffer[face_id] = []
            
            prediction_buffer[face_id].append(preds)
            if len(prediction_buffer[face_id]) > buffer_size:
                prediction_buffer[face_id].pop(0)
            
            avg_preds = np.mean(prediction_buffer[face_id], axis=0)
            pred_index = np.argmax(avg_preds)
            
            # Top 3
            top3_idx = np.argsort(avg_preds)[-3:][::-1]
            
            results.append({
                "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                "primary_emotion": class_names[pred_index],
                "primary_confidence": float(avg_preds[pred_index] * 100),
                "top3": [{"emotion": class_names[i], "confidence": float(avg_preds[i] * 100)} for i in top3_idx],
                "all_probabilities": {class_names[i]: float(avg_preds[i] * 100) for i in range(len(class_names))}
            })
        
        return jsonify({
            "success": True,
            "faces_detected": len(results),
            "faces": results
        })
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Local development
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))