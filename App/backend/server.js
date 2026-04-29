const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

// กำหนดพอร์ตสำหรับ Backend (เช่น 5000)
const PORT = process.env.PORT || 5000;
const app = express();

// Middleware
// อนุญาตให้ Frontend (Vite มักจะรันพอร์ต 5173) เรียกใช้งาน API ได้
app.use(cors()); 
app.use(express.json());

// ---------------------------------------------------------
// การตั้งค่า Multer สำหรับจัดการไฟล์อัปโหลด
// ---------------------------------------------------------

// สร้างโฟลเดอร์ uploads/ ถ้ายังไม่มี
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

// กำหนดที่เก็บไฟล์และชื่อไฟล์ที่จะบันทึก
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir); // เก็บไว้ในโฟลเดอร์ uploads
    },
    filename: function (req, file, cb) {
        // ตั้งชื่อไฟล์ใหม่โดยใช้วันที่-เวลา ป้องกันชื่อไฟล์ซ้ำกัน
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    }
});

// กรองให้รับเฉพาะไฟล์รูปภาพ (เพื่อความปลอดภัย)
const imageFilter = (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
        cb(null, true);
    } else {
        cb(new Error('รองรับเฉพาะไฟล์รูปภาพเท่านั้น!'), false);
    }
};

const upload = multer({ 
    storage: storage,
    fileFilter: imageFilter,
    limits: { fileSize: 5 * 1024 * 1024 } // จำกัดขนาดไฟล์ไม่เกิน 5MB
});

// ---------------------------------------------------------
// API Endpoints
// ---------------------------------------------------------

// API เริ่มต้นสำหรับเช็คว่า Server ทำงานปกติหรือไม่
app.get('/', (req, res) => {
    res.send('Mulberry Defect Backend is running!');
});

// [Step 1] API Endpoint สำหรับรับรูปภาพเพื่อนำไปทำนาย (Predict)
// ใช้ upload.single('image') หมายถึงรับไฟล์เดียวที่ส่งมาในชื่อฟิลด์ 'image'
app.post('/api/predict', upload.single('image'), (req, res) => {
    try {
        // ตรวจสอบว่ามีการอัปโหลดไฟล์มาหรือไม่
        if (!req.file) {
            return res.status(400).json({ 
                status: 'error', 
                message: 'กรุณาอัปโหลดรูปภาพ' 
            });
        }

        console.log(`รับไฟล์รูปภาพแล้ว: ${req.file.filename}`);

        // TODO: [Step 2] เราจะนำไฟล์รูปภาพไปประมวลผลด้วย AI Model ที่นี่
        
        // สำหรับ Step 1 เราจะตอบกลับแบบ Mock data ไปก่อนเพื่อทดสอบระบบ
        res.json({
            status: 'success',
            message: 'อัปโหลดรูปภาพสำเร็จ (เตรียมส่งเข้า AI ใน Step ต่อไป)',
            file_path: `/uploads/${req.file.filename}`
        });

    } catch (error) {
        console.error('เกิดข้อผิดพลาด:', error);
        res.status(500).json({ 
            status: 'error', 
            message: 'เกิดข้อผิดพลาดในการประมวลผลไฟล์เซิร์ฟเวอร์' 
        });
    }
});

// เริ่มรัน Server
app.listen(PORT, () => {
    console.log(`🚀 Server กำลังรันอยู่ที่ http://localhost:${PORT}`);
});