from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import sqlite3
from datetime import datetime

app = FastAPI(title="Student AI App")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained AI model
try:
    model = joblib.load('student_model.pkl')
except FileNotFoundError:
    print("⚠️ Warning: student_model.pkl not found. Run train_model.py first.")
    model = None

# Initialize SQLite Database for Progress Tracking
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, study_hours REAL, attendance REAL, 
                  prev_score REAL, assignments REAL, predicted_score REAL, level TEXT)''')
    conn.commit()
    conn.close()

init_db()  # Run once when server starts

class StudentData(BaseModel):
    study_hours: float
    attendance: float
    prev_score: float
    assignments: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_performance(data: StudentData):
    if model is None:
        return {"error": "Model not loaded"}
        
    # Get prediction from AI
    features = np.array([[data.study_hours, data.attendance, data.prev_score, data.assignments]])
    score = round(max(0, min(100, model.predict(features)[0])), 1)

    # Determine Level & Color
    if score < 50:
        level, color = "At Risk", "#ef4444"
    elif score < 75:
        level, color = "Average", "#f59e0b"
    else:
        level, color = "Excellent", "#10b981"

    # SMART RECOMMENDATION ENGINE
    weaknesses = []
    if data.study_hours < 5: weaknesses.append("Increase weekly study time to at least 5 hours")
    if data.attendance < 70: weaknesses.append("Improve class attendance (currently below 70%)")
    if data.prev_score < 50: weaknesses.append("Review foundational concepts from previous tests")
    if data.assignments < 3: weaknesses.append("Complete more practice assignments")

    if not weaknesses:
        rec = "Excellent performance! Challenge yourself with advanced problems or peer tutoring."
    elif len(weaknesses) == 1:
        rec = f"Focus area: {weaknesses[0]}."
    else:
        rec = "Multiple areas need attention: " + "; ".join(weaknesses[:2]) + "."

    # SAVE TO DATABASE (Progress Tracking)
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("INSERT INTO history (timestamp, study_hours, attendance, prev_score, assignments, predicted_score, level) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), data.study_hours, data.attendance, data.prev_score, data.assignments, score, level))
    conn.commit()
    conn.close()

    return {
        "score": score,
        "level": level,
        "color": color,
        "recommendation": rec
    }

# New endpoint to fetch history for the frontend
@app.get("/history")
def get_history():
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC LIMIT 10")
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows