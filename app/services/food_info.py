# app/services/food_info.py
import pandas as pd
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

# Load and normalize once at import time
_DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "food_nutrition.csv"
_FOOD_DF = pd.read_csv(_DATA_PATH)
_FOOD_DF["key"] = _FOOD_DF["food"].str.lower().str.strip()
_FOOD_KEYS = _FOOD_DF["key"].tolist()


@lru_cache(maxsize=256)
def get_food_match(query: str) -> str | None:
    """Return the best matching food key or None."""
    q = query.lower().strip()
    matches = get_close_matches(q, _FOOD_KEYS, n=1, cutoff=0.6)
    return matches[0] if matches else None


def get_food_info(name: str) -> dict | None:
    """Return nutrition info for the given food name or None."""
    key = get_food_match(name)
    if not key:
        return None
    row = _FOOD_DF[_FOOD_DF["key"] == key].iloc[0]
    return {
        "name": row["food"],
        "calories": row["Caloric Value"],
        "fat_g": row["Fat"],
        "saturated_fat_g": row["Saturated Fats"],
        "carbs_g": row["Carbohydrates"],
        "sugars_g": row["Sugars"],
        "protein_g": row["Protein"],
        "fiber_g": row["Dietary Fiber"],
    }
