### knn_matcher.py (in app/models)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from app.utils.parser import extract_items
import random


class KNNFitnessRecommender:
    def __init__(self, data_path: str, n_neighbors: int = 10):
        self.data_path = data_path
        self.n_neighbors = n_neighbors
        self.pipeline = None
        self.data = None
        self.nn_model = None

    def load_and_prepare_data(self):
        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        df.drop_duplicates(inplace=True)
        df.dropna(thresh=6, inplace=True)
        self.data = df

        features = [
            "sex",
            "age",
            "height",
            "weight",
            "bmi",
            "hypertension",
            "diabetes",
            "level",
            "fitness_goal",
            "fitness_type",
        ]

        column_transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    ["age", "height", "weight", "bmi"],
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    [
                        "sex",
                        "hypertension",
                        "diabetes",
                        "level",
                        "fitness_goal",
                        "fitness_type",
                    ],
                ),
            ]
        )

        self.pipeline = Pipeline(steps=[("preprocessor", column_transformer)])
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine")

        X = self.pipeline.fit_transform(self.data[features])
        self.nn_model.fit(X)

    def match(self, user_input: dict, days_per_week: int = 4):
        features = [
            "sex",
            "age",
            "height",
            "weight",
            "bmi",
            "hypertension",
            "diabetes",
            "level",
            "fitness_goal",
            "fitness_type",
        ]
        X_input = self.pipeline.transform(pd.DataFrame([user_input], columns=features))
        distances, indices = self.nn_model.kneighbors(X_input)

        matched_rows = self.data.iloc[indices[0]].drop_duplicates(
            subset=["exercises", "equipment", "diet"]
        )
        if len(matched_rows) < days_per_week:
            matched_rows = self.data.sample(n=days_per_week)
        else:
            matched_rows = matched_rows.sample(n=days_per_week)

        program = []

        for i, (_, row) in enumerate(matched_rows.iterrows()):
            program.append(
                {
                    "day": i + 1,
                    "exercises": [e.strip() for e in row["exercises"].split(",")],
                    "equipment": [e.strip() for e in row["equipment"].split(",")],
                    "nutrition": {
                        "vegetables": extract_items(row["diet"], "Vegetables"),
                        "proteins": extract_items(row["diet"], "Protein Intake"),
                        "juice": extract_items(row["diet"], "Juice"),
                    },
                    "tips": row["recommendation"],
                }
            )

        return {
            "profile": user_input,
            "days_per_week": days_per_week,
            "program": program,
        }
