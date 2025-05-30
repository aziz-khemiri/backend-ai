from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import requests
from app.services.blood_pressure_service import predict_hypertension

router = APIRouter()

NODE_HYPERTENSION_URL = "http://localhost:5000/api/save1"

class BloodPressureInput(BaseModel):
    age: int
    systolic_pressure: float
    diastolic_pressure: float
    user_id: str  # pour associer la prédiction à un utilisateur

@router.post("/blood_pressure/predict")
async def predict(data: BloodPressureInput):
    input_data = np.array([[ 
        data.age, data.systolic_pressure, data.diastolic_pressure
    ]])

    result = predict_hypertension(input_data)

    payload = {
        "userId": data.user_id,
        "input": data.dict(exclude={"user_id"}),
        "result": result
    }

    try:
        res = requests.post(NODE_HYPERTENSION_URL, json=payload)
        res.raise_for_status()
    except Exception as e:
        print("❌ Failed to send hypertension prediction:", e)

    return {"prediction": result}
