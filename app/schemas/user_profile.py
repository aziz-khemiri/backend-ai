from pydantic import BaseModel, Field
from typing import Literal, Optional


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
