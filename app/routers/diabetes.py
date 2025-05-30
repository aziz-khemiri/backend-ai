from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import requests
from app.services.diabetes_service import predict_diabetes

router = APIRouter()

NODE_DIABETES_URL = "http://localhost:5000/api/save"

class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int
    user_id: str  # Ajouté pour lier à l'utilisateur

@router.post("/diabetes/predict")
async def predict(data: DiabetesInput):
    input_data = np.array([[ 
        data.pregnancies, data.glucose, data.blood_pressure,
        data.skin_thickness, data.insulin, data.bmi,
        data.diabetes_pedigree, data.age
    ]])

    result = predict_diabetes(input_data)

    # Envoyer vers Node.js
    payload = {
        "userId": data.user_id,
        "input": data.dict(exclude={"user_id"}),
        "result": result
    }

    try:
        res = requests.post(NODE_DIABETES_URL, json=payload)
        res.raise_for_status()
    except Exception as e:
        print("❌ Failed to send prediction:", e)

    return {"prediction": result}
