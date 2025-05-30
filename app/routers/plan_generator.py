from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Dict, Union
import pandas as pd
import re
import random
import requests

router = APIRouter()

# =====================================================================
# 📦 DATA LOADING & CLEANING
# =====================================================================
DATA_DIR = "app/datasets"
user_data = pd.read_csv(f"{DATA_DIR}/gym_data_cleaned.csv")
exercise_raw = pd.read_csv(f"{DATA_DIR}/weightlifting_721_workouts.csv")

exercise_raw["exercise_clean"] = (
    exercise_raw["Exercise Name"]
    .str.lower()
    .str.replace(r"[^a-z0-9 ]", "", regex=True)
    .str.strip()
)

best_set = (
    exercise_raw.sort_values(["exercise_clean", "Weight"], ascending=[True, False])
    .drop_duplicates(subset=["exercise_clean"])[
        [
            "Workout Name",
            "exercise_clean",
            "Exercise Name",
            "Weight",
            "Reps",
            "Set Order",
        ]
    ]
    .rename(columns={"Set Order": "Sets"})
    .reset_index(drop=True)
)

bool_cols = {"Hypertension": "hypertension", "Diabetes": "diabetes"}
for col in bool_cols:
    user_data[col] = (
        user_data[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "true"])
        .astype(int)
    )


# =====================================================================
# 🧠 Models & Mappings
# =====================================================================
class UserProfile(BaseModel):
    sex: str
    age: int
    height: float
    weight: float
    level: str
    goal: str
    target_weight: float
    days_per_week: int
    bmi: float | None = None
    hypertension: Union[str, bool] = "No"
    diabetes: Union[str, bool] = "No"
    userId: str | None = None  # ID MongoDB optionnel


class ExerciseDetail(BaseModel):
    exercise: str
    sets: int
    reps: int
    weight: float


class GeneratedPlan(BaseModel):
    weekly_plan: Dict[str, List[ExerciseDetail]]
    equipment: List[str]
    recommendation: str
    diet: str | None = None


MUSCLE_GROUPS = {
    "Chest": ["chest", "bench", "dip", "press"],
    "Back": ["back", "row", "pull", "chin", "lat"],
    "Legs": ["squat", "leg", "lunge", "deadlift"],
    "Shoulders": ["shoulder", "overhead", "military", "delt"],
    "Arms": ["bicep", "tricep", "curl", "pushdown"],
}


# =====================================================================
# 🔍 Similarity Matching
# =====================================================================
def _bool(v: Union[str, bool]) -> bool:
    return str(v).strip().lower() in {"yes", "true", "1"}


def top_k_similar(profile: UserProfile, k: int = 5) -> pd.DataFrame:
    df = user_data.copy()
    df["sim"] = 0.0
    df = df[(df["Sex"].str.lower() == profile.sex.lower())]
    if df.empty:
        return pd.DataFrame()
    df["sim"] += abs(df["Age"] - profile.age) / 5.0
    df["sim"] += abs(df["Weight"] - profile.weight) / 10.0
    df["sim"] += abs(df["Height"] * 100 - profile.height) / 10.0
    for csv_col, prof_attr in bool_cols.items():
        df["sim"] += (df[csv_col] != int(_bool(getattr(profile, prof_attr)))).astype(
            int
        ) * 2
    return df.sort_values("sim").head(k)


# =====================================================================
# 🏋️ Utilities
# =====================================================================
def get_group_exercises(group: str, count: int) -> List[str]:
    keywords = MUSCLE_GROUPS.get(group, [])
    filtered = best_set[
        best_set["exercise_clean"].apply(lambda x: any(kw in x for kw in keywords))
    ]
    return filtered.drop_duplicates("exercise_clean").sample(min(count, len(filtered)))


def detail_list(rows: pd.DataFrame) -> List[ExerciseDetail]:
    seen = set()
    details = []
    for _, row in rows.iterrows():
        clean = re.sub(r"[^a-z0-9 ]", "", str(row["Exercise Name"]).lower())
        if clean in seen:
            continue
        seen.add(clean)
        details.append(
            ExerciseDetail(
                exercise=row["Exercise Name"],
                sets=int(row["Sets"] if row["Sets"] else 3),
                reps=int(row["Reps"] if row["Reps"] else 10),
                weight=float(row["Weight"] if row["Weight"] else 0.0),
            )
        )
    return details


# =====================================================================
# 📨 Send to Node.js Backend
# =====================================================================
NODE_BACKEND_URL = "http://localhost:5000/api/plan/save"


def send_to_nodejs(user_id: str, plan: dict):
    try:
        payload = {
            "userId": user_id,
            "weekly_plan": plan["weekly_plan"],
            "equipment": plan["equipment"],
            "recommendation": plan["recommendation"],
            "diet": plan["diet"],
        }
        res = requests.post(NODE_BACKEND_URL, json=payload)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print("❌ Failed to send to Node.js backend:", e)
        return None


# =====================================================================
# 🚀 Main Endpoint
# =====================================================================
@router.post("/plan/generate", response_model=GeneratedPlan)
def generate_plan(profile: UserProfile):
    df_sim = top_k_similar(profile)
    equipment: set[str] = set()
    recommendation = ""
    diet = ""

    for _, row in df_sim.iterrows():
        equipment.update(
            [eq.strip() for eq in str(row["Equipment"]).split(",") if eq.strip()]
        )
        if not recommendation:
            recommendation = str(row["Recommendation"])
        if not diet:
            diet = str(row["Diet"])

    selected_groups = random.sample(
        list(MUSCLE_GROUPS.keys()), k=min(profile.days_per_week, len(MUSCLE_GROUPS))
    )
    random.shuffle(selected_groups)

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekly_plan: Dict[str, List[ExerciseDetail]] = {}

    for i in range(profile.days_per_week):
        group = selected_groups[i % len(selected_groups)]
        day = day_names[i % 7]
        exercises = get_group_exercises(group, 5)
        weekly_plan[day] = detail_list(exercises)

    result = {
        "weekly_plan": weekly_plan,
        "equipment": sorted(equipment),
        "recommendation": recommendation,
        "diet": diet or None,
    }

    if profile.userId:
        send_to_nodejs(profile.userId, result)

    return result
