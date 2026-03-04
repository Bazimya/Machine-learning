from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import joblib
import numpy as np


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("house_price_model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API Running"}

@app.post("/predict")
def predict(size: float, bedrooms: int, bathrooms: int, location_score: int):
    
    input_data = np.array([[size, bedrooms, bathrooms, location_score]])
    prediction = model.predict(input_data)

    return {
        "predicted_price_million_RWF": round(prediction[0], 2)
    }