import base64
import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)

# 1. แก้ CORS ให้รับ Request จากทุกที่ (หรือใส่ URL ของ Netlify ทีหลัง)
CORS(app, resources={r"/*": {"origins": "*"}})

# ===== CONFIG =====
# ตรวจสอบว่าไฟล์โมเดลอยู่ในโฟลเดอร์ model จริงๆ
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "final_model2_fix2.h5")

class_names = ['Fungal leaf', 'Good leaf', 'Insect-eaten leaf', 'Not mulberry leaf']

def build_model():
    # ... (ส่วน build_model ของคุณเหมือนเดิม) ...
    # ใส่โค้ดส่วน build_model เดิมของคุณที่นี่
    img_height = 224
    img_width = 224
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
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4)
    ])
    return model

# โหลด Model
try:
    model = build_model()
    model.load_weights(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def preprocess(img, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Mulberry AI API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    # ... (ส่วน predict ของคุณเหมือนเดิม) ...
    data = request.get_json()
    if "image_base64" not in data:
        return jsonify({"error": "No image provided"}), 400
    try:
        image_data = data["image_base64"].split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        input_data = preprocess(img)
        prediction = model.predict(input_data)
        predicted_index = int(np.argmax(prediction, axis=1)[0])
        prediction_prob = tf.nn.softmax(prediction, axis=1).numpy()
        
        return jsonify({
            "predicted_class_name": class_names[predicted_index],
            "confidence": float(prediction_prob[0][predicted_index]),
            "prediction": prediction.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 2. แก้ให้รันตาม Port ที่ Server กำหนด
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)