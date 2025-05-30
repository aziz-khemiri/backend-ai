import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Charger et entraîner le modèle depuis le CSV au démarrage
csv_path = os.path.join(os.path.dirname(__file__), "../datasets/diabetes.csv")
df = pd.read_csv(csv_path)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


def predict_diabetes(input_data: np.ndarray) -> str:
    """
    Prend un tableau numpy (shape: [1,8]) et retourne la prédiction ("Diabetic" ou "Not Diabetic").
    """
    prediction = model.predict(input_data)[0]
    return "Diabetic" if prediction == 1 else "Not Diabetic"
