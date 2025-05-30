from fastapi import FastAPI
from app.routers import chat, plan_generator, diabetes,blood_pressure_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Medical Chat API")

# Inclure les routeurs séparément
app.include_router(chat.router, prefix="/api")
app.include_router(plan_generator.router, prefix="/api")
app.include_router(diabetes.router, prefix="/api")
app.include_router(blood_pressure_router.router, prefix="/api", tags=["Blood Pressure Prediction"])

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origin(s) if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "MedChat API is running 🚀"}
