from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import requests
import os
from app.services.blood_pressure_service import predict_hypertension

router = APIRouter()

# ✅ URL dynamique avec fallback local
NODE_HYPERTENSION_URL = os.getenv("NODE_HYPERTENSION_URL", "http://localhost:5000/api/save1")

# 📦 Modèle attendu en POST
class BloodPressureInput(BaseModel):
    age: int
    systolic_pressure: float
    diastolic_pressure: float
    user_id: str  # pour associer la prédiction à un utilisateur

# 📡 Route API publique
@router.post("/blood_pressure/predict")
async def predict(data: BloodPressureInput):
    try:
        # 📊 Données sous forme de tableau
        input_data = np.array([[ 
            data.age, data.systolic_pressure, data.diastolic_pressure
        ]])

        # 🤖 Appel modèle de prédiction
        result = predict_hypertension(input_data)

        # 📤 Préparation de l’envoi vers backend Node.js
        payload = {
            "userId": data.user_id,
            "input": data.dict(exclude={"user_id"}),
            "result": result
        }

        # 🌐 Envoi POST vers Node.js
        response = requests.post(NODE_HYPERTENSION_URL, json=payload)
        response.raise_for_status()

        return { "prediction": result }

    except requests.RequestException as e:
        print("❌ Failed to send hypertension prediction:", e)
        raise HTTPException(status_code=502, detail="Failed to send prediction to Node backend.")
    except Exception as e:
        print("❌ Internal error:", e)
        raise HTTPException(status_code=500, detail="Internal server error.")
