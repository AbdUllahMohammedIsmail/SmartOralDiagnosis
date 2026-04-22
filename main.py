from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import faiss
import io
import json
from PIL import Image

app = FastAPI()

model = tf.keras.models.load_model('model.h5')
index = faiss.read_index('oral_cases.index')

with open('case_ids.json', 'r') as f:
    cases_list = json.load(f)

@app.get("/")
def read_root():
    return {"status": "AI Server is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # أ. معالجة الصورة
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB').resize((224, 224))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # ب. استخراج الميزات (Embedding)
    embedding = model.predict(img_array)
    
    # ج. البحث في FAISS
    distances, indices = index.search(embedding.astype('float32'), k=1)
    
    # د. تحديد الحالة الأقرب من القائمة
    matched_index = indices[0][0]
    if matched_index < len(cases_list):
        matched_case_path = cases_list[matched_index] 
        
        disease_category = matched_case_path.split('\\')[0] 
    else:
        matched_case_path = "Unknown"
        disease_category = "Unknown"

    return {
        "matched_case": matched_case_path,
        "disease_category": disease_category,  #عشان مصطفى العبيط يرتاح
        "confidence": float(1 / (1 + distances[0][0]))
    }
