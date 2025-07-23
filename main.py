from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI()
model = tf.keras.models.load_model('energy_model.h5')
scaler = joblib.load('scaler.save')

class Features(BaseModel):
    data: list
@app.post('/predict')
def predict(features: Features):
    try:
        X=np.array(features.data).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        return {"prediction": float(pred[0,0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))