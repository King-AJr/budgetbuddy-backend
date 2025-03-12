from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import auth
from app.routes import finance
from app.core.config import get_settings
import firebase_admin

firebase_admin.initialize_app()

settings = get_settings()

app = FastAPI(title=settings.PROJECT_NAME)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(finance.router, prefix="/api", tags=["api"])

@app.get("/")
async def root():
    return {"message": "Welcome to BudgetBuddy-Genius"} 