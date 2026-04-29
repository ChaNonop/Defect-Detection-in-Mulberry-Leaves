from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import numpy as np
import tensorflow as tf
from utils.image_process import preprocess_image

# สร้างแอปพลิเคชัน FastAPI
app = FastAPI(title="Mulberry Leaf Defect Detection API")

# ตั้งค่า CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด Model เมื่อแอปพลิเคชันเริ่มต้น
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "mulberry_leaf_classification.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Not found Model file at: {MODEL_PATH}")

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# รายชื่อ Class ตามลำดับตัวอักษร
CLASS_NAMES = ["Healthy", "Rust", "Spot"]

@app.get("/")
async def root():
    """
    Health Check Endpoint
    """
    return {"message": "Welcome to Mulberry Leaf Defect Detection API"}

@app.post("/api/upload/")
async def predict_leaf(file: UploadFile = File(...)):
    """
    Endpoint สำหรับรับไฟล์รูปภาพและทำนายโรคใบหม่อน
    """
    # 1. ตรวจสอบนามสกุลไฟล์ที่อนุญาต
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail="รูปแบบไฟล์ไม่ถูกต้อง รองรับเฉพาะ .jpg, .jpeg, และ .png เท่านั้น"
        )
    
    try:
        # 2. อ่านไฟล์รูปภาพเป็น Bytes
        content = await file.read()
        
        # 3. ทำ Pre-processing
        processed_image = preprocess_image(content)
        
        # 4. ทำนายผลด้วย Model
        predictions = model.predict(processed_image)
        
        # 5. ประมวลผลลัพธ์
        scores = predictions[0]
        top_index = np.argmax(scores)
        
        predicted_class = CLASS_NAMES[top_index]
        confidence = float(scores[top_index]) * 100
        
        # สร้าง Dict สำหรับคะแนนทั้งหมด
        all_scores = {
            CLASS_NAMES[i]: float(scores[i]) * 100 
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            "status": "success",
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "all_scores": {k: round(v, 2) for k, v in all_scores.items()},
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")

if __name__ == "__main__":
    # รันเซิร์ฟเวอร์ด้วย Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)