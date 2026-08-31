from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List
import joblib
import numpy as np
import json
import os
import re
import ast
import operator
import requests
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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

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

class DoubtInput(BaseModel):
    question: str
    board: str = ""
    student_class: str = ""
    stream: str = "None"
    weak_subjects: List[str] = []

# ---------- OFFLINE TUTOR ----------
def eval_expr(expr):
    allowed = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.operand))
        raise ValueError("not allowed")
    return _eval(ast.parse(expr, mode='eval').body)

CANNED = {
    "photosynthesis": "🔬 Photosynthesis is how green plants make food using sunlight, water and CO2, releasing oxygen. Equation: 6CO2 + 6H2O + sunlight → C6H12O6 + 6O2. It happens in chloroplasts containing chlorophyll.",
    "gravity": "⚛️ Gravity is the force that pulls objects toward each other. Earth's gravity pulls everything toward its center with acceleration ~9.8 m/s². That's why apples fall down!",
    "noun": "📚 A noun is a naming word — person (teacher), place (Delhi), thing (book), or idea (happiness). Example: 'Riya went to school.' → Riya and school are nouns.",
    "fraction": "📐 A fraction represents a part of a whole, written as numerator/denominator (e.g., 3/4 = 3 parts out of 4). To add fractions, make denominators equal first!",
    "percentage": "📐 Percentage means 'per 100'. Formula: (Part ÷ Whole) × 100. Example: 45/60 = 0.75 → 75%.",
    "evaporation": "🔬 Evaporation is when liquid water turns into vapor due to heat. Example: puddles drying in the sun. It's part of the water cycle!",
}

def fallback_tutor(q):
    if re.search(r"\d", q) and re.search(r"[\+\-\*/×÷]", q):
        cleaned = q.lower().replace("x", "*").replace("×", "*").replace("÷", "/")
        m = re.findall(r"[\d\.\+\-\*/\(\)\s]+", cleaned)
        for part in sorted(m, key=len, reverse=True):
            part = part.strip()
            if len(part) >= 3:
                try:
                    val = eval_expr(part)
                    return f"📐 Calculation: {part} = {round(val, 4)}\n\nTip: Follow BODMAS order — Brackets, Orders (powers), Division/Multiplication, Addition/Subtraction."
                except Exception:
                    continue
    ql = q.lower()
    for key, ans in CANNED.items():
        if key in ql:
            return ans
    if any(w in ql for w in ["math", "algebra", "geometry"]):
        return "📐 For Math doubts: (1) Understand the formula, (2) Solve one example step-by-step, (3) Practice 5 similar problems. Tell me the exact topic (e.g., quadratic equations) and I'll guide you!"
    if any(w in ql for w in ["science", "physics", "chemistry", "biology"]):
        return "🔬 For Science doubts: read the concept, draw a diagram, and explain it in your own words. Tell me the exact topic and I'll break it down simply!"
    if any(w in ql for w in ["english", "grammar", "essay"]):
        return "📚 For English: read the question twice, note keywords, and answer in simple sentences. Tell me the exact topic (grammar, essay, comprehension)!"
    return "🤔 I'm your offline tutor (add a free Gemini API key for full AI answers!). Try asking: 'What is photosynthesis?', 'Solve 12*8+4', 'What is a noun?', or tell me your exact topic and I'll give a study plan."

@app.post("/ask")
async def ask(data: DoubtInput):
    context = (f"Student profile: {data.board} {data.student_class}, stream: {data.stream}, "
               f"weak subjects: {', '.join(data.weak_subjects) if data.weak_subjects else 'none'}.")
    if GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
            prompt = (f"You are a friendly personal tutor. {context} "
                      f"Explain simply, suitable for their class, with one example. Keep under 200 words.\n\n"
                      f"Student doubt: {data.question}")
            r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
            out = r.json()
            answer = out["candidates"][0]["content"]["parts"][0]["text"]
            return {"answer": answer, "source": "gemini"}
        except Exception:
            return {"answer": fallback_tutor(data.question), "source": "offline"}
    return {"answer": fallback_tutor(data.question), "source": "offline"}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

@app.get("/manifest.json")
async def manifest():
    return FileResponse("manifest.json", media_type="application/json")

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