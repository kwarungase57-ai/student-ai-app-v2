from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict
import joblib
import numpy as np
import json
import os
from database import SessionLocal, StudentRecord, init_db

app = FastAPI(title="AI Student Analyzer & Timetable Generator")
init_db()

MODEL_PATH = "student_model.pkl"
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

EMOJIS = {
    "Math": "📐", "Science": "🔬", "English": "📚", "Hindi": "🖋️",
    "Social Studies": "🌍", "Physics": "⚛️", "Chemistry": "🧪",
    "Computer Science": "💻", "Computer Applications": "💻",
    "Marathi": "📖", "Accountancy": "🧾", "Business Studies": "💼",
    "Economics": "📈", "History": "🏛️", "Political Science": "🗳️",
    "Geography": "🗺️"
}

def emoji_for(subject):
    return EMOJIS.get(subject, "📘")

class StudentInput(BaseModel):
    board: str
    student_class: str
    stream: str = "None"
    study_hours: float
    subject_scores: Dict[str, float]

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

def get_advice(subject, score):
    e = emoji_for(subject)
    if score < 40:
        return f"{e} {subject} Critical: start from basics. Watch video lectures daily + solve NCERT examples"
    if score < 60:
        return f"{e} {subject} Weak: practice 30–45 mins daily, revise notes and solve exercises"
    if score < 75:
        return f"{e} {subject} Moderate: solve previous year papers and focus on tricky topics"
    return None

def generate_timetable(study_hours, subject_scores):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    total_gap = sum(max(0, 100 - s) for s in subject_scores.values())
    if total_gap == 0:
        weights = {s: 1/len(subject_scores) for s in subject_scores}
    else:
        weights = {s: max(0, 100 - sc) / total_gap for s, sc in subject_scores.items()}

    daily_hours = study_hours / 7
    timetable = []
    for i, day in enumerate(days):
        tasks = []
        if day == "Sunday":
            tasks.append("📝 Weekly Review & Test")
            tasks.append("🧘 Rest & Light Reading")
        else:
            for subj, w in weights.items():
                hrs = round(daily_hours * w, 1)
                if hrs >= 0.5:
                    tasks.append(f"{emoji_for(subj)} {subj} ({hrs}h)")
            if i % 2 == 0:
                tasks.append("📘 General Revision (0.5h)")
            tasks.append("✏️ Homework / Assignments")
        timetable.append({"day": day, "tasks": tasks})
    return timetable

@app.post("/predict")
async def predict(data: StudentInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")

    try:
        scores = list(data.subject_scores.values())
        if not scores:
            raise ValueError("No subject scores provided")

        class_num = int(data.student_class.replace("Class ", ""))
        avg = sum(scores) / len(scores)
        mn = min(scores)
        mx = max(scores)

        features = np.array([[class_num, data.study_hours, avg, mn, mx]])
        prediction = model.predict(features)[0]
        confidence = float(max(model.predict_proba(features)[0]))

        weak_subjects = [s for s, sc in data.subject_scores.items() if sc < 75]
        recommendations = []
        for subj, sc in sorted(data.subject_scores.items(), key=lambda x: x[1]):
            advice = get_advice(subj, sc)
            if advice:
                recommendations.append(advice)

        if data.study_hours < 14:
            recommendations.append("⏰ Low study hours! Aim for at least 2-3 hours/day")
        if not weak_subjects:
            recommendations.append("🌟 Excellent scores! Focus on advanced problems & revision")

        timetable = generate_timetable(data.study_hours, data.subject_scores)

        db = SessionLocal()
        try:
            record = StudentRecord(
                board=data.board,
                student_class=data.student_class,
                stream=data.stream,
                study_hours=data.study_hours,
                avg_score=round(avg, 1),
                subject_scores=json.dumps(data.subject_scores),
                performance_level=prediction
            )
            db.add(record)
            db.commit()
        finally:
            db.close()

        return {
            "performance_level": prediction,
            "confidence": round(confidence * 100, 1),
            "average_score": round(avg, 1),
            "weak_subjects": weak_subjects,
            "recommendations": recommendations,
            "timetable": timetable
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/history")
async def get_history():
    db = SessionLocal()
    try:
        records = db.query(StudentRecord).order_by(StudentRecord.timestamp.desc()).limit(20).all()
        return [{
            "timestamp": r.timestamp.isoformat(),
            "student_class": r.student_class,
            "avg_score": r.avg_score
        } for r in records]
    finally:
        db.close()