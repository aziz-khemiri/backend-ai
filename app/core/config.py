from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openrouter_api_key: str
    model_name: str = "deepseek-chat"

    class Config:
        env_file = ".env"


settings = Settings()
