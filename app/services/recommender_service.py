from app.models.knn_matcher import KNNFitnessRecommender
import os

MODEL_PATH = os.path.join("app", "datasets", "gym_data_cleaned.csv")

recommender = KNNFitnessRecommender(data_path=MODEL_PATH)
recommender.load_and_prepare_data()
