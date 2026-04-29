from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """
    ทำ Pre-processing รูปภาพก่อนนำเข้า AI Model
    1. แปลง bytes เป็น PIL Image
    2. Resize เป็น target_size (224x224)
    3. แปลงเป็น RGB (กรณีเป็น RGBA หรือ Grayscale)
    4. แปลงเป็น numpy array และ Normalize (หารด้วย 255.0)
    5. เพิ่มมิติสำหรับ Batch (1, 224, 224, 3)
    """
    # 1. แปลง bytes เป็น PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # 2. แปลงเป็น RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # 3. Resize
    image = image.resize(target_size)
    
    # 4. แปลงเป็น numpy array และ Normalize
    image_array = np.array(image) / 255.0
    
    # 5. เพิ่มมิติสำหรับ Batch
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array
