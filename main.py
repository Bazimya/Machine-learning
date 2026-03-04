from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

# Create FastAPI app
app = FastAPI()

# Allow frontend requests (safe for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tell FastAPI where HTML files are located
templates = Jinja2Templates(directory="templates")

# Load trained ML model
model = joblib.load("house_price_model.pkl")


# 🔥 ROOT ROUTE (Serves Frontend)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 🔥 PREDICTION ROUTE
@app.post("/predict")
def predict(size: float, bedrooms: int, bathrooms: int, location_score: int):
    try:
        # Prepare input data
        input_data = np.array([[size, bedrooms, bathrooms, location_score]])

        # Make prediction
        prediction = model.predict(input_data)

        return {
            "predicted_price_million_RWF": round(float(prediction[0]), 2)
        }

    except Exception as e:
        return {"error": str(e)}