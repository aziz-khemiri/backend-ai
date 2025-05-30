from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import requests
import os
from app.services.diabetes_service import predict_diabetes

router = APIRouter()

# Utiliser une variable d'environnement pour l'URL du backend Node.js
NODE_DIABETES_URL = os.getenv("NODE_DIABETES_URL", "http://localhost:5000/api/save")

# Modèle de requête
class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int
    user_id: str  # Lier la prédiction à l'utilisateur

# 🔁 Fonction isolée pour sauvegarder la prédiction dans Node.js
def save_prediction_to_node(url: str, user_id: str, input_data: dict, result: str):
    payload = {
        "userId": user_id,
        "input": input_data,
        "result": result
    }
    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
    except requests.RequestException as e:
        print("❌ Failed to send prediction:", e)
        raise HTTPException(status_code=502, detail="Failed to send prediction to Node backend.")

# Route d’API
@router.post("/diabetes/predict")
async def predict(data: DiabetesInput):
    try:
        input_array = np.array([[
            data.pregnancies, data.glucose, data.blood_pressure,
            data.skin_thickness, data.insulin, data.bmi,
            data.diabetes_pedigree, data.age
        ]])

        result = predict_diabetes(input_array)

        # Envoi des résultats vers Node.js
        save_prediction_to_node(
            NODE_DIABETES_URL,
            data.user_id,
            data.dict(exclude={"user_id"}),
            result
        )

        return {"prediction": result}

    except Exception as e:
        print("❌ Internal error:", e)
        raise HTTPException(status_code=500, detail="Internal server error.")
