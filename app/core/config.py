from pydantic_settings import BaseSettings
from functools import lru_cache
import os

MODEL_NAME = os.getenv("MODEL_NAME")

class Settings(BaseSettings):
    PROJECT_NAME: str = "BudgetBuddy-Genius"
    FIREBASE_CREDENTIALS: str
    FIREBASE_WEB_API_KEY: str
    GROQ_API_KEY: str
    MODEL_NAME: str = MODEL_NAME
    PROJECT_ID: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()