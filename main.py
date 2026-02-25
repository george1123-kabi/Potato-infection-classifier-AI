# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf 
from pathlib import Path
from keras.layers import Dense
from keras.saving import register_keras_serializable
import uvicorn
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL=None

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "potato.h5"

class SafeDense(Dense):
    def __init__(self, **kwargs):
        # We pop the offending key before passing it to the real Dense layer
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

# 2. When loading the model, tell Keras to use SafeDense instead of Dense
def safe_load_model(path):
    try:

        model = tf.keras.models.load_model(
            path,
            custom_objects={"Dense": SafeDense},
            compile=False 
        )
        return model
    except TypeError as e:
        print(f"Surgery needed: {e}")
        # If the standard load fails, we return None and look at logs
        return None
try:
    MODEL = safe_load_model(str(MODEL_PATH))
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")

CLASS_NAMES=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def read_file_as_img(data)->np.ndarray:
    image=Image.open(BytesIO(data)).convert('RGB')
    image=image.resize((256,256),Image.Resampling.BILINEAR)
    return image



@app.post("/predict")
async def predict(
    file:UploadFile=File(...)
):
    global MODEL
    if MODEL is None:
        return {"error": "Model not loaded on server. Check logs."}
    image=read_file_as_img(await file.read())

    img_batch=np.expand_dims(image,0)
    
    prediction=MODEL.predict(img_batch)

    predicted_img=CLASS_NAMES[np.argmax(prediction[0])]
    confidence_level=np.max(prediction[0])
    actions={
    "Potato___Early_blight":"Prune infected lower leaves and apply a copper-based fungicide to stop the spread of 'target-spot' lesions.",
    "Potato___Late_blight":"Immediately rogue (destroy) infected plants and apply aggressive systemic fungicides to prevent total crop loss.",
    "Potato___Healthy":"The plant is healthy"
  }
    


    predictions={
        "class_name":predicted_img,
        "confidence":float(confidence_level),
        "action":actions[predicted_img]
        
    }

    return predictions
    
    

if __name__=="__main__":
    uvicorn.run(app,port=8000,host="0.0.0.0")