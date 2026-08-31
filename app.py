from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
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


class StudentInput(BaseModel):
    student_class: str
    career_stream: str
    study_hours: float
    math_score: float
    science_score: float
    english_score: float


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


def generate_timetable(study_hours, math, science, english, stream):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    scores = {"Math": math, "Science": science, "English": english}
    total_gap = sum(max(0, 100 - s) for s in scores.values())

    if total_gap == 0:
        weights = {"Math": 0.33, "Science": 0.33, "English": 0.34}
    else:
        weights = {subj: max(0, 100 - score) / total_gap for subj, score in scores.items()}

    stream_tasks = {
        "Science": ["Physics Practice", "Chemistry Revision"],
        "Commerce": ["Accounting Problems", "Business Studies"],
        "Arts": ["History Notes", "Political Science"],
        "Vocational": ["Practical Skills", "Workshop Practice"]
    }
    extra_tasks = stream_tasks.get(stream, ["General Revision"])
    daily_hours = study_hours / 7
    timetable = []

    for i, day in enumerate(days):
        tasks = []
        if day == "Sunday":
            tasks.append("📝 Weekly Review & Test")
            tasks.append("🧘 Rest & Light Reading")
        else:
            for subj, weight in weights.items():
                hrs = round(daily_hours * weight, 1)
                if hrs >= 0.5:
                    emoji = {"Math": "📐", "Science": "🔬", "English": "📚"}[subj]
                    tasks.append(f"{emoji} {subj} ({hrs}h)")
            if i % 2 == 0 and extra_tasks:
                tasks.append(f"📘 {extra_tasks[i % len(extra_tasks)]} (0.5h)")
            tasks.append("✏️ Homework / Assignments")
        timetable.append({"day": day, "tasks": tasks})
    return timetable


@app.post("/predict")
async def predict(data: StudentInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")

    try:
        class_num = int(data.student_class.replace("Class ", ""))
        stream_map = {"Science": 1, "Commerce": 2, "Arts": 3, "Vocational": 4}
        stream_num = stream_map.get(data.career_stream, 0)

        features = np.array([[
            class_num, stream_num, data.study_hours,
            data.math_score, data.science_score, data.english_score
        ]])

        prediction = model.predict(features)[0]
        confidence = float(max(model.predict_proba(features)[0]))

        weak_subjects = []
        recommendations = []
        thresholds = {"Mathematics": data.math_score, "Science": data.science_score, "English": data.english_score}

        advice_map = {
            "Mathematics": [
                (40, "📐 Critical: Start with basics. Use Khan Academy + NCERT solved examples daily"),
                (60, "📐 Weak: Practice 10 problems/day focusing on weak topics. Use formula sheets"),
                (75, "📐 Moderate: Solve previous year papers. Focus on speed & accuracy"),
            ],
            "Science": [
                (40, "🔬 Critical: Rebuild concepts from scratch. Watch video lectures + draw diagrams"),
                (60, "🔬 Weak: Make flashcards for key concepts. Practice numericals separately"),
                (75, "🔬 Moderate: Focus on application-based questions and lab practicals"),
            ],
            "English": [
                (40, "📚 Critical: Read 1 chapter/story daily. Build vocabulary with word lists"),
                (60, "📚 Weak: Practice writing summaries. Read editorials for comprehension"),
                (75, "📚 Moderate: Practice essay writing. Analyze past paper patterns"),
            ]
        }

        for subj, score in thresholds.items():
            if score < 75:
                weak_subjects.append(subj)
                for threshold, advice in advice_map[subj]:
                    if score < threshold:
                        recommendations.append(advice)
                        break

        if data.study_hours < 14:
            recommendations.append("⏰ Low study hours! Aim for at least 2-3 hours/day for improvement")
        if not weak_subjects:
            recommendations.append("🌟 Excellent diagnostic scores! Focus on advanced problems & revision")

        timetable = generate_timetable(
            data.study_hours, data.math_score, data.science_score,
            data.english_score, data.career_stream
        )

        db = SessionLocal()
        try:
            record = StudentRecord(
                student_class=data.student_class,
                career_stream=data.career_stream,
                study_hours=data.study_hours,
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
            "math_score": r.math_score,
            "science_score": r.science_score
        } for r in records]
    finally:
        db.close()  