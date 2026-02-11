import base64
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import os
import tensorflow as tf
from tensorflow.keras import layers, models

app = Flask(__name__)

# อนุญาต CORS
CORS(app, resources={r"/predict": {"origins": "https://g3weds.consolutechcloud.com"}})

# ===== CONFIG =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "final_model2_fix2.h5")

# ชื่อคลาส (4 คลาส)
class_names = ['Fungal leaf', 'Good leaf', 'Insect-eaten leaf', 'Not mulberry leaf']

img_height = 224
img_width = 224

# === สร้างสถาปัตยกรรมเหมือนตอนเทรน ===
def build_model(num_classes=4, input_shape=(224, 224, 3), dropout_rate=0.3):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes)  # 4 คลาส
    ])
    return model

# โหลดน้ำหนักโมเดล
model = build_model(num_classes=4)
model.load_weights(MODEL_PATH)
print("✅ Model weights (4 classes) loaded successfully!")

# ====== ฟังก์ชัน Preprocess ======
def preprocess(img, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0   # ✅ normalize เป็นช่วง [0,1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    
# ====== ROUTES ======
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"}), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    data = request.get_json()

    if "image_base64" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # แปลง base64 -> Image
        image_data = data["image_base64"].split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(image_data)))

        # Preprocess
        input_data = preprocess(img)

        # Predict
        prediction = model.predict(input_data)
        predicted_index = int(np.argmax(prediction, axis=1)[0])

        # Softmax -> Probabilities
        prediction_prob = tf.nn.softmax(prediction, axis=1).numpy()

        # Class & Confidence
        predicted_class_name = class_names[predicted_index]
        confidence = float(prediction_prob[0][predicted_index])

        return jsonify({
            "prediction": prediction.tolist(),
            "prediction_prob": prediction_prob.tolist(),
            "predicted_index": predicted_index,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4200, debug=True)
