import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Charger et entraîner le modèle depuis le CSV
csv_path = os.path.join(os.path.dirname(__file__), "../datasets/blood_pressure_large_dataset.csv")
df = pd.read_csv(csv_path)

# Convertir la colonne hypertension en 0/1
df['hypertension'] = df['hypertension'].map({'No': 0, 'Yes': 1})

X = df[["age", "systolic_pressure", "diastolic_pressure"]]
y = df["hypertension"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def predict_hypertension(input_data: np.ndarray) -> str:
    """
    Prend un tableau numpy (shape: [1,3]) et retourne la prédiction ("Hypertensive" ou "Normal").
    """
    prediction = model.predict(input_data)[0]
    return "Hypertensive" if prediction == 1 else "Normal"