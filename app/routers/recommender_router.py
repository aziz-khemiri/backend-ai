# routers/recommender_router.py
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal, Optional
from app.services.recommender_service import recommender

router = APIRouter()


class UserProfile(BaseModel):
    sex: Literal["Male", "Female"]
    age: int
    height: float
    weight: float
    hypertension: Literal["Yes", "No"]
    diabetes: Literal["Yes", "No"]
    bmi: float
    level: str
    fitness_goal: str
    fitness_type: str
    days_per_week: Optional[int] = Field(default=3, ge=1, le=7)


@router.post("/recommend")
def recommend_plan(user: UserProfile):
    response = recommender.match(user.dict(), days_per_week=user.days_per_week)
    return response
