const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
// [Step 2] นำเข้า TensorFlow.js สำหรับ Node.js และแพ็กเกจสำหรับ TFLite
const tf = require('@tensorflow/tfjs-node');
const tflite = require('@tensorflow/tfjs-tflite');

const PORT = process.env.PORT || 5000;
const app = express();

app.use(cors()); 
app.use(express.json());

// ---------------------------------------------------------
// โหลด AI Model (TFLite)
// ---------------------------------------------------------
let model;

// ฟังก์ชันโหลดโมเดลตอนเริ่มเปิดเซิร์ฟเวอร์
async function loadModel() {
    try {
        // ชี้ไปที่ไฟล์ .tflite ที่คุณเตรียมไว้
        const modelPath = path.join(__dirname, 'model', 'mulberry_leaf_classification.tflite');
        console.log('⏳ กำลังโหลด AI Model (.tflite)...');
        
        // สำหรับ Node.js การอ่านไฟล์เป็น Buffer แล้วแปลงให้ TFLite จะเสถียรที่สุด
        const modelBuffer = fs.readFileSync(modelPath);
        model = await tflite.loadTFLiteModel(new Uint8Array(modelBuffer));
        
        console.log('✅ โหลด AI Model (.tflite) สำเร็จและพร้อมใช้งาน!');
    } catch (error) {
        console.error('❌ ไม่สามารถโหลด AI Model ได้ (กรุณาเช็คว่ามีไฟล์ .tflite ในโฟลเดอร์ model):', error.message);
    }
}
loadModel(); // เรียกใช้งานทันทีเมื่อเริ่มเซิร์ฟเวอร์

// ---------------------------------------------------------
// การตั้งค่า Multer สำหรับอัปโหลด
// ---------------------------------------------------------
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir); 
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    }
});

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
    limits: { fileSize: 5 * 1024 * 1024 } 
});

// ---------------------------------------------------------
// API Endpoints
// ---------------------------------------------------------
app.get('/', (req, res) => {
    res.send('Mulberry Defect Backend is running! 🌿');
});

// [Step 2] ปรับปรุง API เพื่อประมวลผลรูปภาพด้วย AI
app.post('/api/predict', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ status: 'error', message: 'กรุณาอัปโหลดรูปภาพ' });
        }

        if (!model) {
            return res.status(503).json({ status: 'error', message: 'AI Model ยังโหลดไม่เสร็จ หรือมีปัญหาในการโหลด' });
        }

        console.log(`เริ่มวิเคราะห์รูปภาพ: ${req.file.filename}`);

        // ใช้ tf.tidy() เพื่อจัดการลบ Tensor ออกจาก Memory อัตโนมัติเมื่อทำงานเสร็จ (Best Practice)
        const predictionResult = tf.tidy(() => {
            // 1. อ่านไฟล์รูปภาพเป็น Buffer
            const imageBuffer = fs.readFileSync(req.file.path);
            
            // 2. แปลงรูปภาพเป็น Tensor (3 ช่องสี RGB)
            const tfImage = tf.node.decodeImage(imageBuffer, 3);
            
            // 3. ปรับขนาดรูปให้ตรงกับที่ Model เทรนมา (ส่วนใหญ่เป็น 224x224 หรือตามที่คุณเทรนมา)
            const resizedImage = tf.image.resizeBilinear(tfImage, [224, 224]);
            
            // 4. ขยายมิติให้เป็น Batch (1, 224, 224, 3) และ Normalize (สำหรับ TFLite มักจะ /255.0)
            const expandedImage = resizedImage.expandDims(0).toFloat().div(255.0);
            
            // 5. ทำนายผลลัพธ์
            const predictions = model.predict(expandedImage);
            return predictions.dataSync(); // ดึงข้อมูลออกมาเป็น Array ทันที
        });

        // 6. ตีความผลลัพธ์
        // ลำดับ Index (0, 1) ต้องตรงกับตอนที่คุณเทรนโมเดลมานะครับ
        const CLASS_NAMES = ['Defect', 'Good']; 
        
        // หา Class ที่มีค่าเปอร์เซ็นต์ความน่าจะเป็นสูงสุด
        const maxScore = Math.max(...predictionResult);
        const classIndex = predictionResult.indexOf(maxScore);
        const condition = CLASS_NAMES[classIndex];

        console.log(`✅ ผลวิเคราะห์: ${condition} (${(maxScore * 100).toFixed(2)}%)`);

        // ลบไฟล์รูปภาพทิ้งหลังประมวลผลเสร็จเพื่อประหยัดพื้นที่เซิร์ฟเวอร์ (ทางเลือก)
        // fs.unlinkSync(req.file.path);

        // 7. ส่งผลลัพธ์กลับไปให้ Frontend
        res.json({
            status: 'success',
            condition: condition,
            confidence: maxScore,
            file_path: `/uploads/${req.file.filename}`
        });

    } catch (error) {
        console.error('เกิดข้อผิดพลาดในการรันโมเดล:', error);
        res.status(500).json({ status: 'error', message: 'เกิดข้อผิดพลาดในการประมวลผล AI' });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Server กำลังรันอยู่ที่ http://localhost:${PORT}`);
});