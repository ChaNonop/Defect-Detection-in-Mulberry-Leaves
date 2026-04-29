from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# สร้างแอปพลิเคชัน FastAPI
app = FastAPI(title="Mulberry Leaf Defect Detection API")

# ตั้งค่า CORS Middleware เพื่ออนุญาตให้ Frontend (React) เรียกใช้งาน API ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ในโหมด Production ควรเปลี่ยนเป็น URL ของ Frontend จริงๆ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# สร้างโฟลเดอร์ temp ชั่วคราวหากยังไม่มี เพื่อไว้เก็บไฟล์รูปที่ผู้ใช้อัปโหลด
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
async def root():
    """
    Health Check Endpoint เพื่อทดสอบว่า API ทำงานปกติหรือไม่
    """
    return {"message": "Welcome to Mulberry Leaf Defect Detection API"}

@app.post("/api/upload/")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint สำหรับรับไฟล์รูปภาพจาก Frontend
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
        # 2. บันทึกไฟล์ลงในโฟลเดอร์ temp
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # หมายเหตุ: ใน Step 2 เราจะนำไฟล์ที่เซฟไว้นี้ไปผ่านการ Pre-processing และโยนเข้า AI Model
        
        # 3. ส่ง Response กลับไปหา Frontend
        return {
            "status": "success",
            "message": "อัปโหลดไฟล์รูปภาพสำเร็จ",
            "filename": file.filename,
            "filepath": file_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการอัปโหลดไฟล์: {str(e)}")

if __name__ == "__main__":
    # รันเซิร์ฟเวอร์ด้วย Uvicorn ที่พอร์ต 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)