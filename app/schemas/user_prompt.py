from pydantic import BaseModel


class UserPrompt(BaseModel):
    age: int
    sex: str
    height: int
    weight: float
    level: str
    goal: str

    def to_prompt(self):
        return (
            f"I am a {self.age}-year-old {self.sex.lower()} with a height of {self.height} cm and "
            f"a weight of {self.weight} kg. My primary goal is {self.goal}, and I am a {self.level} at the gym."
        )
