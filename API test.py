import requests
import os

# ตั้งค่า URL ของ API
URL = "http://localhost:8000/api/upload/"

def test_prediction(image_path):
    """
    ฟังก์ชันสำหรับส่งรูปภาพไปทดสอบที่ API
    """
    if not os.path.exists(image_path):
        print(f"Not found image file at: {image_path}")
        return

    print(f"Sending image {image_path} to analyze...")
    
    try:
        # เปิดไฟล์รูปภาพแบบ Binary
        with open(image_path, "rb") as img_file:
            files = {"file": (os.path.basename(image_path), img_file, "image/jpeg")}
            
            # ส่ง POST Request
            response = requests.post(URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("\nAnalyze Success!")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']}%")
                print("-" * 30)
                print("All Details:")
                for leaf_class, score in result['all_scores'].items():
                    print(f"- {leaf_class}: {score}%")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
    except Exception as e:
        print(f"Connection Error: {str(e)}")

if __name__ == "__main__":
    # ระบุชื่อไฟล์รูปภาพที่คุณต้องการทดสอบ
    test_image = "test_leaf.jpg" 
    
    # ถ้ายังไม่มีไฟล์รูป ให้ลองรัน Health Check ก่อน
    try:
        health_check = requests.get("http://localhost:8000/")
        print(f"API Status: {health_check.json()['message']}")
        
        # รันเทสรูปภาพ
        test_prediction(test_image)
    except Exception as e:
        print("Cannot connect to API, please check if main.py is running.")
        print(str(e))