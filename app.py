from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os
from database import SessionLocal, StudentRecord, init_db

# Initialize FastAPI & Database
app = FastAPI(title="AI Student Performance Predictor")
init_db()

# Load Trained Model
MODEL_PATH = "student_model.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found! Run train_model.py first.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Input Schema (Must match training features exactly)
class StudentInput(BaseModel):
    study_hours: float
    attendance: float
    math_score: float
    science_score: float
    english_score: float

# Serve Frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

# Prediction + Recommendation Endpoint
@app.post("/predict")
async def predict(data: StudentInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    
    try:
        # 1. Prepare Features
        features = np.array([[
            data.study_hours, 
            data.attendance, 
            data.math_score, 
            data.science_score, 
            data.english_score
        ]])
        
        # 2. Predict Performance Level
        prediction = model.predict(features)[0]
        confidence = float(max(model.predict_proba(features)[0]))
        
        # 3. Generate Personalized Recommendations
        recommendations = []
        weak_subjects = []
        
        if data.math_score < 60:
            weak_subjects.append("Mathematics")
            recommendations.append("📐 Math: Practice 30 mins daily using Khan Academy")
        if data.science_score < 60:
            weak_subjects.append("Science")
            recommendations.append(" Science: Focus on conceptual diagrams & flashcards")
        if data.english_score < 60:
            weak_subjects.append("English")
            recommendations.append("📚 English: Read 1 article daily & summarize it")
        if data.study_hours < 5:
            recommendations.append("⏰ Time: Increase study time to 5+ hours/week")
        if data.attendance < 75:
            recommendations.append("🎓 Attendance: Aim for 85%+ to improve retention")
            
        if not recommendations:
            recommendations.append("🌟 Excellent progress! Maintain your current routine.")
        
        # 4. Save Record to Database
        db = SessionLocal()
        try:
            record = StudentRecord(
                study_hours=data.study_hours,
                attendance=data.attendance,
                math_score=data.math_score,
                science_score=data.science_score,
                english_score=data.english_score,
                performance_level=prediction
            )
            db.add(record)
            db.commit()
        finally:
            db.close()
        
        # 5. Return Response
        return {
            "performance_level": prediction,
            "confidence": round(confidence * 100, 1),
            "weak_subjects": weak_subjects,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# History Endpoint for Charts
@app.get("/history")
async def get_history():
    db = SessionLocal()
    try:
        records = db.query(StudentRecord).order_by(
            StudentRecord.timestamp.desc()
        ).limit(20).all()
        
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "performance_level": r.performance_level,
                "math_score": r.math_score,
                "study_hours": r.study_hours
            } 
            for r in records
        ]
    finally:
        db.close()