import React, { useState, useRef } from 'react';

const RECOMMENDATIONS = {
  "Healthy": "ใบหม่อนสมบูรณ์ดี ดูแลรักษาความสะอาดของแปลงและให้น้ำตามปกติเพื่อป้องกันการเกิดโรค",
  "Rust": "พบโรคราสนิม: แนะนำให้ตัดแต่งกิ่งและทำลายใบที่เป็นโรคทันทีเพื่อป้องกันการแพร่กระจาย หลีกเลี่ยงการให้น้ำโดนใบ และอาจใช้สารชีวภัณฑ์หรือสารเคมีควบคุมเชื้อรา",
  "Spot": "พบโรคใบจุด: แนะนำให้เก็บใบที่ร่วงและตัดใบที่เป็นโรคไปทำลาย ลดความหนาแน่นของกิ่งก้านเพื่อให้แสงแดดส่องถึงและอากาศถ่ายเทได้ดียิ่งขึ้น"
};

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Handle file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('กรุณาอัปโหลดไฟล์รูปภาพ (.jpg, .png)');
      return;
    }
    setImage(file);
    setResult(null);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);
    setIsCameraOpen(false);
  };

  // Drag and Drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  // Camera handlers
  const startCamera = async () => {
    setIsCameraOpen(true);
    setImage(null);
    setPreview(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("ไม่สามารถเปิดกล้องได้ กรุณาใช้วิธีอัปโหลดไฟล์แทน");
      setIsCameraOpen(false);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      const dataUrl = canvas.toDataURL('image/jpeg');
      setPreview(dataUrl);
      
      fetch(dataUrl)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
          setImage(file);
        });

      stopCamera();
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
    setIsCameraOpen(false);
  };

  const resetApp = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Fetch API Function
  const analyzeImage = async () => {
    if (!image) return;
    
    setLoading(true);
    setResult(null);
    
    const formData = new FormData();
    formData.append('file', image);
    
    try {
      const response = await fetch('http://localhost:8000/api/upload/', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('การเชื่อมต่อเซิร์ฟเวอร์ล้มเหลว');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Analysis Error:", error);
      alert("เกิดข้อผิดพลาดในการวิเคราะห์: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card">
      <div className="header">
        <h1 className="title">Mulberry AI</h1>
        <p className="subtitle">ระบบปัญญาประดิษฐ์ตรวจจับโรคใบหม่อน</p>
      </div>

      {/* Camera View */}
      {isCameraOpen && (
        <div className="camera-container">
          <video ref={videoRef} className="camera-video" playsInline />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>
      )}

      {/* Image Preview */}
      {preview && !isCameraOpen && (
        <div className="preview-container">
          <img src={preview} alt="Preview" className="preview-image" />
          {!loading && !result && (
            <button className="remove-btn" onClick={resetApp}>✕</button>
          )}
        </div>
      )}

      {/* Upload Zone */}
      {!preview && !isCameraOpen && (
        <div 
          className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <svg className="upload-icon pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <div>
            <p style={{ fontWeight: 600, marginBottom: '0.25rem', color: 'var(--text-primary)' }}>
              กดเพื่ออัปโหลด หรือลากไฟล์มาที่นี่
            </p>
            <p className="subtitle">รองรับไฟล์ .jpg, .png</p>
          </div>
          <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            style={{ display: 'none' }}
          />
        </div>
      )}

      {/* Action Buttons */}
      <div className="actions-group">
        {isCameraOpen ? (
          <>
            <button className="btn-primary" onClick={capturePhoto}>
              ถ่ายรูป
            </button>
            <button className="btn-secondary" onClick={stopCamera}>
              ยกเลิก
            </button>
          </>
        ) : (
          <>
            {!preview && (
              <button className="btn-secondary" onClick={startCamera}>
                <svg style={{width:'20px', height:'20px'}} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                เปิดกล้อง
              </button>
            )}
          </>
        )}
      </div>

      {/* Analyze Button */}
      {preview && !isCameraOpen && !result && (
        <button 
          className="btn-primary" 
          onClick={analyzeImage} 
          disabled={loading}
        >
          {loading ? (
            <>
              <div className="loading-spinner" />
              <span>กำลังประมวลผล...</span>
            </>
          ) : (
            'เริ่มวิเคราะห์โรค'
          )}
        </button>
      )}

      {/* Results Section */}
      {result && (
        <div className="fadeIn" style={{ marginTop: '1rem' }}>
          <div style={{ 
            background: result.prediction === 'Healthy' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
            padding: '1.25rem', 
            borderRadius: 'var(--radius-md)',
            marginBottom: '1.5rem',
            border: `1px solid ${result.prediction === 'Healthy' ? 'var(--primary)' : '#ef4444'}`,
            textAlign: 'center',
            boxShadow: '0 4px 12px rgba(0,0,0,0.05)'
          }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
              ผลการตรวจสอบ
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 700, color: result.prediction === 'Healthy' ? 'var(--primary-dark)' : '#991b1b' }}>
              {result.prediction === 'Healthy' ? 'ปกติ (Healthy)' : 
               result.prediction === 'Rust' ? 'โรคราสนิม (Rust)' : 'โรคใบจุด (Spot)'}
            </div>
            <div style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
              ความมั่นใจ: <strong>{result.confidence}%</strong>
            </div>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-primary)' }}>
              รายละเอียดความน่าจะเป็น:
            </h3>
            {Object.entries(result.all_scores).map(([className, score]) => (
              <div key={className} style={{ marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem', color: 'var(--text-primary)' }}>
                  <span style={{ fontWeight: className === result.prediction ? 600 : 400 }}>
                    {className === 'Healthy' ? 'ปกติ (Healthy)' : 
                     className === 'Rust' ? 'โรคราสนิม (Rust)' : 'โรคใบจุด (Spot)'}
                  </span>
                  <span style={{ fontWeight: className === result.prediction ? 600 : 400 }}>{score}%</span>
                </div>
                <div style={{ height: '10px', background: 'rgba(0,0,0,0.05)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
                  <div style={{ 
                    width: `${score}%`, 
                    height: '100%', 
                    background: className === 'Healthy' ? 'var(--primary)' : '#ef4444',
                    borderRadius: 'var(--radius-full)',
                    transition: 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
                  }} />
                </div>
              </div>
            ))}
          </div>

          <div style={{ 
            background: 'rgba(255,255,255,0.4)', 
            padding: '1.25rem', 
            borderRadius: 'var(--radius-md)',
            fontSize: '0.9rem',
            lineHeight: '1.5',
            color: 'var(--text-secondary)',
            border: '1px solid var(--glass-border)',
            marginBottom: '1.5rem',
            boxShadow: '0 2px 8px rgba(0,0,0,0.02)'
          }}>
            <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.25rem' }}>คำแนะนำ:</strong> 
            {RECOMMENDATIONS[result.prediction]}
          </div>

          <button className="btn-secondary" onClick={resetApp}>
            วิเคราะห์ภาพใหม่
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
