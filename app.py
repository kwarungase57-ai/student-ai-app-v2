from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# 1. Load Model (Only once at startup)
MODEL_PATH = "student_model.pkl"  # <--- CHANGE THIS to your actual model filename
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 2. Define Input Data Structure (Match your model's inputs exactly)
class StudentInput(BaseModel):
    study_hours: float
    attendance: float
    previous_score: float
    # Add/Remove fields here to match YOUR model exactly!

# 3. Serve Frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

# 4. Prediction Endpoint
@app.post("/predict")
async def predict(data: StudentInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    
    try:
        # Convert input to array [Must match training order!]
        features = np.array([[
            data.study_hours, 
            data.attendance, 
            data.previous_score
            # Add more variables here if your model needs them
        ]])
        
        prediction = model.predict(features)
        
        return {
            "prediction": int(prediction[0]),
            "message": "Prediction successful"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))