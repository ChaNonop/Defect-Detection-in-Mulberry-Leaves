const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
// กำหนด Port โดยดึงจาก Environment Variable (Render จะเป็นคนกำหนดให้) หรือใช้ 3000 ตอนรันในเครื่อง
const PORT = process.env.PORT || 3000;

// ==========================================
// 1. Middlewares
// ==========================================
// อนุญาตให้ Frontend เรียกใช้ API ข้ามโดเมนได้ (จำเป็นตอนพัฒนาที่เครื่องเรา)
app.use(cors());
// แปลงข้อมูลที่ส่งมาเป็น JSON ให้อ่านง่ายๆ
app.use(express.json()); 


// ==========================================
// 2. API Routes (ส่วนหลังบ้าน เผื่ออนาคต)
// ==========================================
// API เช็คสถานะเซิร์ฟเวอร์
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Backend is running smoothly! 🍃' });
});

// [ตัวอย่าง] API สำหรับรับข้อมูลผลตรวจใบหม่อน เผื่ออนาคตนำไปบันทึกลง Database
app.post('/api/save-log', (req, res) => {
  const { isDefected, confidence } = req.body;
  
  // TODO: ในอนาคตสามารถเขียนโค้ดบันทึกลง MongoDB หรือฐานข้อมูลอื่นๆ ตรงนี้ได้
  console.log(`[Log Received] ใบหม่อนมีตำหนิหรือไม่: ${isDefected}, ความมั่นใจ: ${confidence}%`);
  
  res.json({ success: true, message: 'Log saved (Mock)' });
});


// ==========================================
// 3. Frontend Serving (สำหรับตอน Deploy จริง)
// ==========================================
// กำหนด path ชี้ไปยังโฟลเดอร์ที่ Build มาจาก React (ปกติจะเป็นโฟลเดอร์ dist)
// สมมติว่าโครงสร้างโฟลเดอร์คือ:
// /project-root
//   /backend
//     server.js
//   /frontend
//     /dist (ได้จากคำสั่ง npm run build)
const frontendPath = path.join(__dirname, '../frontend/dist');
app.use(express.static(frontendPath));

// Fallback Route: หากผู้ใช้เข้า URL ย่อยอื่นๆ ให้ส่งหน้า index.html ของ React ไปจัดการต่อ (รองรับ React Router)
app.get('*', (req, res) => {
  res.sendFile(path.join(frontendPath, 'index.html'));
});


// ==========================================
// 4. Start Server
// ==========================================
app.listen(PORT, () => {
  console.log(`🚀 Server is running on port ${PORT}`);
});