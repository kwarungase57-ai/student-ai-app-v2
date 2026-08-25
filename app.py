from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os
from database import SessionLocal, StudentRecord, init_db

app = FastAPI(title="AI Student Performance Predictor")
init_db()

MODEL_PATH = "student_model.pkl"
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ✅ UPDATED: Added student_class as first field
class StudentInput(BaseModel):
    student_class: str
    study_hours: float
    attendance: float
    math_score: float
    science_score: float
    english_score: float

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

@app.post("/predict")
async def predict(data: StudentInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    
    try:
        # Encode class to number for model (6->6, 10->10, etc.)
        class_num = int(data.student_class.replace("Class ", ""))
        
        features = np.array([[
            class_num,           # 1. Class
            data.study_hours,    # 2. Hours
            data.attendance,     # 3. Attendance
            data.math_score,     # 4. Math
            data.science_score,  # 5. Science
            data.english_score   # 6. English
        ]])
        
        prediction = model.predict(features)[0]
        confidence = float(max(model.predict_proba(features)[0]))
        
        recommendations = []
        weak_subjects = []
        if data.math_score < 60:
            weak_subjects.append("Mathematics")
            recommendations.append(" Math: Practice 30 mins daily")
        if data.science_score < 60:
            weak_subjects.append("Science")
            recommendations.append("🔬 Science: Use visual diagrams")
        if data.english_score < 60:
            weak_subjects.append("English")
            recommendations.append("📚 English: Read daily articles")
        if data.study_hours < 5:
            recommendations.append("⏰ Increase study time to 5+ hrs/week")
        if data.attendance < 75:
            recommendations.append("🎓 Aim for 85%+ attendance")
        if not recommendations:
            recommendations.append("🌟 Excellent! Maintain routine.")
        
        db = SessionLocal()
        try:
            record = StudentRecord(
                student_class=data.student_class,
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
        
        return {
            "performance_level": prediction,
            "confidence": round(confidence * 100, 1),
            "weak_subjects": weak_subjects,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/history")
async def get_history():
    db = SessionLocal()
    try:
        records = db.query(StudentRecord).order_by(StudentRecord.timestamp.desc()).limit(20).all()
        return [{"timestamp": r.timestamp.isoformat(), "student_class": r.student_class, "math_score": r.math_score} for r in records]
    finally:
        db.close()