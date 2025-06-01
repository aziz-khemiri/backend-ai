from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import requests
import os
from app.services.blood_pressure_service import predict_hypertension

router = APIRouter()

# ✅ URL Render de production
NODE_HYPERTENSION_URL = os.getenv("NODE_HYPERTENSION_URL", "https://back-6jw0.onrender.com/api/save1")

class BloodPressureInput(BaseModel):
    age: int
    systolic_pressure: float
    diastolic_pressure: float
    user_id: str

# 🔁 Fonction isolée pour envoyer une prédiction vers Node.js
def save_prediction_to_node(url: str, user_id: str, input_data: dict, result: str):
    payload = {
        "userId": user_id,
        "input": input_data,
        "result": result
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print("❌ Failed to send hypertension prediction:", e)
        raise HTTPException(status_code=502, detail="Failed to send prediction to Node backend.")

@router.post("/blood_pressure/predict")
async def predict(data: BloodPressureInput):
    try:
        input_data = np.array([[
            data.age, data.systolic_pressure, data.diastolic_pressure
        ]])

        result = predict_hypertension(input_data)

        save_prediction_to_node(
            NODE_HYPERTENSION_URL,
            data.user_id,
            data.dict(exclude={"user_id"}),
            result
        )

        return {"prediction": result}

    except Exception as e:
        print("❌ Internal error:", e)
        raise HTTPException(status_code=500, detail="Internal server error.")
