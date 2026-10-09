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
import time
from database import SessionLocal, StudentRecord, init_db

app = FastAPI(title="AI Study Abroad Platform")
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
if GEMINI_API_KEY:
    print("✅ Gemini API key detected")
else:
    print("⚠️ No GEMINI_API_KEY found — using offline tutor")

GEMINI_MODELS = ["gemini-2.6-flash", "gemini-2.6-flash-lite", "gemini-flash-latest",
                 "gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-2.5-flash",
                 "gemini-2.0-flash-lite", "gemini-2.0-flash"]

class StudentInput(BaseModel):
    student_name: str = "Student"
    level: str = "Class 12"       # "Class 12" or "Graduation"
    stream: str = "Science"
    percentage: float
    study_hours: float = 20

class DoubtInput(BaseModel):
    question: str
    board: str = ""
    student_class: str = ""
    stream: str = "None"
    weak_subjects: List[str] = []

class TestGenInput(BaseModel):
    board: str = ""
    student_class: str = ""
    items: List[Dict[str, str]] = []

# ---------- SELF-HEALING GEMINI ----------
def gemini_answer(prompt):
    if not GEMINI_API_KEY:
        return None, None
    models_to_try = list(GEMINI_MODELS)
    tried = set()
    for _ in range(4):
        for model_name in models_to_try:
            if model_name in tried:
                continue
            tried.add(model_name)
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
                r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=40)
                if r.status_code == 200:
                    out = r.json()
                    answer = out["candidates"][0]["content"]["parts"][0]["text"]
                    print(f"✅ Gemini answered via {model_name}")
                    return answer, model_name
                print(f"⚠️ Gemini {model_name} error {r.status_code}: {r.text[:250]}")
                if r.status_code == 404:
                    m = re.search(r"(gemini-[a-z0-9.\-]+)", r.text)
                    if m and m.group(1) not in tried:
                        models_to_try.append(m.group(1))
                        print(f"🔁 Discovered newer model: {m.group(1)}")
                elif r.status_code in (503, 429):
                    time.sleep(2)
            except Exception as e:
                print(f"⚠️ Gemini {model_name} failed: {e}")
    return None, None

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
    "photosynthesis": "🔬 Photosynthesis is how green plants make food using sunlight, water and CO2, releasing oxygen. Equation: 6CO2 + 6H2O + sunlight → C6H12O6 + 6O2.",
    "gravity": "⚛️ Gravity pulls objects toward each other with acceleration ~9.8 m/s² on Earth.",
    "noun": "📚 A noun is a naming word — person, place, thing or idea. Example: 'Riya went to school.'",
    "fraction": "📐 A fraction = part/whole (e.g., 3/4). Make denominators equal before adding!",
    "percentage": "📐 Percentage = (Part ÷ Whole) × 100. Example: 45/60 = 75%.",
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
                    return f"📐 Calculation: {part} = {round(val, 4)}"
                except Exception:
                    continue
    ql = q.lower()
    for key, ans in CANNED.items():
        if key in ql:
            return ans
    return "🤔 Try asking: 'What is photosynthesis?', 'Solve 12*8+4', or any study question!"

@app.post("/ask")
async def ask(data: DoubtInput):
    context = (f"Student profile: {data.student_class}, stream: {data.stream}, "
               f"weak subjects: {', '.join(data.weak_subjects) if data.weak_subjects else 'none'}.")
    ql = data.question.lower()
    wants_chapter = any(w in ql for w in ["chapter", "notes", "summary", "text of", "full topic"])
    if wants_chapter:
        prompt = (f"You are a friendly personal tutor. {context} Provide a study summary with: "
                  f"1) Overview 2) Key concepts 3) Formulas 4) One solved example 5) Revision tips. Under 500 words.\n\n"
                  f"Student request: {data.question}")
    else:
        prompt = (f"You are a friendly personal tutor. {context} Explain simply with one example. Under 200 words.\n\n"
                  f"Student doubt: {data.question}")
    answer, model_name = gemini_answer(prompt)
    if answer:
        if wants_chapter:
            answer += "\n\n📕 For official NCERT text: https://ncert.nic.in/textbook.php"
        return {"answer": answer, "source": "gemini"}
    if wants_chapter:
        return {"answer": "📕 Read the official NCERT text free at https://ncert.nic.in/textbook.php. Ask me any topic and I'll explain!", "source": "offline"}
    return {"answer": fallback_tutor(data.question), "source": "offline"}

# ---------- AI TEST GENERATOR ----------
SERVER_QUIZ = {
    "Math": [
        {"q": "15% of 200 = ?", "a": ["20","30","40","25"], "c": 1},
        {"q": "12 × 8 + 4 = ?", "a": ["96","100","104","98"], "c": 1},
        {"q": "Square root of 144 = ?", "a": ["10","11","12","14"], "c": 2},
    ],
    "Science": [
        {"q": "Gas absorbed by plants?", "a": ["Oxygen","Nitrogen","CO2","Hydrogen"], "c": 2},
        {"q": "Unit of force?", "a": ["Joule","Newton","Watt","Pascal"], "c": 1},
        {"q": "Powerhouse of the cell?", "a": ["Nucleus","Mitochondria","Ribosome","Chloroplast"], "c": 1},
    ],
    "English": [
        {"q": "Synonym of 'Happy'?", "a": ["Sad","Joyful","Angry","Tired"], "c": 1},
        {"q": "Plural of 'Child'?", "a": ["Childs","Children","Childes","Child"], "c": 1},
    ],
    "Social": [
        {"q": "First PM of India?", "a": ["Nehru","Gandhi","Patel","Rajendra Prasad"], "c": 0},
        {"q": "Taj Mahal is in?", "a": ["Delhi","Jaipur","Agra","Lucknow"], "c": 2},
    ]
}

def server_quiz_for(subject):
    if re.search(r"math", subject, re.I): return SERVER_QUIZ["Math"]
    if re.search(r"science|physics|chemistry|biology", subject, re.I): return SERVER_QUIZ["Science"]
    if re.search(r"english|hindi|marathi", subject, re.I): return SERVER_QUIZ["English"]
    return SERVER_QUIZ["Social"]

@app.post("/generate_test")
async def generate_test(data: TestGenInput):
    if data.items:
        topics_txt = "; ".join(f"{i.get('subject','')}: {i.get('topic','')}" for i in data.items[:6])
        prompt = (f"You are an expert teacher for {data.student_class}. "
                  f"Create exactly 5 multiple-choice questions on these topics: {topics_txt}. "
                  f"Respond ONLY with a JSON array: "
                  f'[{{"subject":"Math","q":"...","a":["a","b","c","d"],"c":0}}]')
        text, model_name = gemini_answer(prompt)
        if text:
            try:
                m = re.search(r"\[.*\]", text, re.S)
                questions = json.loads(m.group(0))
                if isinstance(questions, list) and questions:
                    print(f"✅ AI test generated via {model_name}")
                    return {"questions": questions, "source": "gemini"}
            except Exception as e:
                print(f"⚠️ Test JSON parse failed: {e}")
    questions = []
    subjects = list({i.get("subject", "Math") for i in data.items}) or ["Math"]
    for s in subjects[:3]:
        questions += [{**q, "subject": s} for q in server_quiz_for(s)][:3]
    return {"questions": questions, "source": "offline"}

# ---------- 🎓 CAREER ENGINE (stream-based) ----------
STREAM_CAREERS = {
    "Science": [
        {"track": "Engineering & Technology", "roles": ["Mechanical / Civil / Computer Engineer", "Robotics Specialist", "ISRO / DRDO Scientist"]},
        {"track": "Software, IT & AI", "roles": ["Software Developer", "AI / ML Engineer", "Cybersecurity Expert"]},
        {"track": "Medical & Healthcare", "roles": ["Doctor (MBBS)", "Pharmacist", "Biotech Scientist"]},
    ],
    "Commerce": [
        {"track": "Commerce, Finance & CA", "roles": ["Chartered Accountant (CA)", "Investment Banker", "Financial Advisor"]},
        {"track": "Business & Management", "roles": ["Entrepreneur", "Marketing Manager", "HR Manager"]},
        {"track": "Data Science & Statistics", "roles": ["Data Scientist", "Statistician", "Business Analyst"]},
    ],
    "Arts": [
        {"track": "Law, Civil Services & Administration", "roles": ["IAS / IPS Officer", "Lawyer / Judge", "Policy Analyst"]},
        {"track": "Humanities, Teaching & Psychology", "roles": ["Teacher / Professor", "Psychologist", "NGO Leader"]},
        {"track": "Media, Writing & Languages", "roles": ["Journalist", "Content Writer", "Translator"]},
    ],
}

GRAD_CAREERS = [
    {"track": "Higher Studies (Masters / PhD)", "roles": ["Research Scholar", "University Lecturer", "Scientist"]},
    {"track": "Government Exams", "roles": ["UPSC Civil Services", "SSC / Banking", "PSU Jobs"]},
    {"track": "Software, IT & AI", "roles": ["Software Developer", "Data Analyst", "AI/ML Engineer"]},
]

def suggest_careers_stream(stream, level, pct):
    base = STREAM_CAREERS.get(stream, STREAM_CAREERS["Science"])
    items = (GRAD_CAREERS + base[:1]) if level == "Graduation" else base
    return [{"track": c["track"], "roles": c["roles"], "match": round(pct)} for c in items[:3]]

# ---------- 🌍 STUDY ABROAD ENGINE ----------
UNIVERSITIES = {
    "UK": {
        "flag": "🇬🇧",
        "dream": [
            {"name": "University of Oxford", "min_gpa": 90, "min_ielts": 7.5, "cost": "high"},
            {"name": "Imperial College London", "min_gpa": 88, "min_ielts": 7.0, "cost": "high"},
            {"name": "University of Cambridge", "min_gpa": 92, "min_ielts": 7.5, "cost": "high"},
        ],
        "match": [
            {"name": "University of Manchester", "min_gpa": 80, "min_ielts": 6.5, "cost": "high"},
            {"name": "King's College London", "min_gpa": 82, "min_ielts": 6.5, "cost": "high"},
            {"name": "University of Edinburgh", "min_gpa": 78, "min_ielts": 6.5, "cost": "medium"},
        ],
        "safety": [
            {"name": "University of Leeds", "min_gpa": 70, "min_ielts": 6.0, "cost": "medium"},
            {"name": "University of Birmingham", "min_gpa": 72, "min_ielts": 6.0, "cost": "medium"},
            {"name": "University of Nottingham", "min_gpa": 68, "min_ielts": 6.0, "cost": "medium"},
        ],
    },
    "USA": {
        "flag": "🇺",
        "dream": [
            {"name": "MIT", "min_gpa": 95, "min_ielts": 7.5, "cost": "high"},
            {"name": "Stanford University", "min_gpa": 93, "min_ielts": 7.5, "cost": "high"},
            {"name": "Harvard University", "min_gpa": 95, "min_ielts": 7.5, "cost": "high"},
        ],
        "match": [
            {"name": "UC Davis", "min_gpa": 82, "min_ielts": 6.5, "cost": "high"},
            {"name": "University of Texas at Austin", "min_gpa": 80, "min_ielts": 6.5, "cost": "medium"},
            {"name": "Purdue University", "min_gpa": 78, "min_ielts": 6.5, "cost": "medium"},
        ],
        "safety": [
            {"name": "Arizona State University", "min_gpa": 70, "min_ielts": 6.0, "cost": "medium"},
            {"name": "University of Kansas", "min_gpa": 68, "min_ielts": 6.0, "cost": "low"},
            {"name": "Iowa State University", "min_gpa": 72, "min_ielts": 6.0, "cost": "low"},
        ],
    },
    "Canada": {
        "flag": "🇨🇦",
        "dream": [
            {"name": "University of Toronto", "min_gpa": 88, "min_ielts": 7.0, "cost": "high"},
            {"name": "McGill University", "min_gpa": 87, "min_ielts": 7.0, "cost": "high"},
            {"name": "University of British Columbia", "min_gpa": 86, "min_ielts": 6.5, "cost": "high"},
        ],
        "match": [
            {"name": "University of Waterloo", "min_gpa": 82, "min_ielts": 6.5, "cost": "medium"},
            {"name": "McMaster University", "min_gpa": 80, "min_ielts": 6.5, "cost": "medium"},
            {"name": "University of Ottawa", "min_gpa": 78, "min_ielts": 6.5, "cost": "medium"},
        ],
        "safety": [
            {"name": "University of Calgary", "min_gpa": 72, "min_ielts": 6.0, "cost": "medium"},
            {"name": "Dalhousie University", "min_gpa": 70, "min_ielts": 6.0, "cost": "low"},
            {"name": "University of Manitoba", "min_gpa": 68, "min_ielts": 6.0, "cost": "low"},
        ],
    },
    "Australia": {
        "flag": "🇦🇺",
        "dream": [
            {"name": "University of Melbourne", "min_gpa": 88, "min_ielts": 7.0, "cost": "high"},
            {"name": "University of Sydney", "min_gpa": 87, "min_ielts": 7.0, "cost": "high"},
            {"name": "Australian National University", "min_gpa": 85, "min_ielts": 6.5, "cost": "high"},
        ],
        "match": [
            {"name": "Monash University", "min_gpa": 80, "min_ielts": 6.5, "cost": "medium"},
            {"name": "University of Queensland", "min_gpa": 78, "min_ielts": 6.5, "cost": "medium"},
            {"name": "University of Adelaide", "min_gpa": 76, "min_ielts": 6.5, "cost": "medium"},
        ],
        "safety": [
            {"name": "University of Wollongong", "min_gpa": 70, "min_ielts": 6.0, "cost": "low"},
            {"name": "RMIT University", "min_gpa": 68, "min_ielts": 6.0, "cost": "low"},
            {"name": "Deakin University", "min_gpa": 66, "min_ielts": 6.0, "cost": "low"},
        ],
    },
    "Germany": {
        "flag": "🇩🇪",
        "dream": [
            {"name": "TU Munich", "min_gpa": 88, "min_ielts": 6.5, "cost": "low"},
            {"name": "RWTH Aachen", "min_gpa": 85, "min_ielts": 6.5, "cost": "low"},
            {"name": "Heidelberg University", "min_gpa": 87, "min_ielts": 6.5, "cost": "low"},
        ],
        "match": [
            {"name": "TU Berlin", "min_gpa": 80, "min_ielts": 6.5, "cost": "low"},
            {"name": "University of Stuttgart", "min_gpa": 78, "min_ielts": 6.0, "cost": "low"},
            {"name": "KIT Karlsruhe", "min_gpa": 80, "min_ielts": 6.5, "cost": "low"},
        ],
        "safety": [
            {"name": "TU Dresden", "min_gpa": 72, "min_ielts": 6.0, "cost": "low"},
            {"name": "University of Hamburg", "min_gpa": 74, "min_ielts": 6.0, "cost": "low"},
            {"name": "TU Darmstadt", "min_gpa": 75, "min_ielts": 6.0, "cost": "low"},
        ],
    },
}

SCHOLARSHIPS = {
    "UK": [
        {"name": "Chevening Scholarship", "value": "Full tuition + living", "min_score": 85, "min_ielts": 6.5},
        {"name": "Commonwealth Scholarship", "value": "Full funding", "min_score": 80, "min_ielts": 6.5},
        {"name": "GREAT Scholarships", "value": "£10,000", "min_score": 75, "min_ielts": 6.0},
    ],
    "USA": [
        {"name": "Fulbright-Nehru Fellowship", "value": "Full funding", "min_score": 85, "min_ielts": 6.5},
        {"name": "Stanford Reliance Fellowship", "value": "Full tuition", "min_score": 90, "min_ielts": 7.0},
        {"name": "Inlaks Shivdasani", "value": "Up to $100K", "min_score": 80, "min_ielts": 6.5},
    ],
    "Canada": [
        {"name": "Vanier Canada Graduate Scholarship", "value": "$50,000/year", "min_score": 85, "min_ielts": 6.5},
        {"name": "Ontario Graduate Scholarship", "value": "$15,000/year", "min_score": 80, "min_ielts": 6.5},
        {"name": "Lester B. Pearson Scholarship", "value": "Full funding", "min_score": 88, "min_ielts": 6.5},
    ],
    "Australia": [
        {"name": "Australia Awards Scholarship", "value": "Full funding", "min_score": 80, "min_ielts": 6.5},
        {"name": "Destination Australia", "value": "$15,000/year", "min_score": 75, "min_ielts": 6.0},
        {"name": "RTP Scholarship", "value": "Full tuition + stipend", "min_score": 85, "min_ielts": 6.5},
    ],
    "Germany": [
        {"name": "DAAD Scholarship", "value": "Full funding", "min_score": 80, "min_ielts": 6.0},
        {"name": "Heinrich Böll Foundation", "value": "€850/month", "min_score": 78, "min_ielts": 6.0},
        {"name": "Konrad Adenauer Stiftung", "value": "€850/month", "min_score": 80, "min_ielts": 6.0},
    ],
}

def check_abroad_eligibility(avg_score, ielts_band, countries, budget):
    budget_order = {"low": 1, "medium": 2, "high": 3}
    user_budget = budget_order.get(budget, 2)
    score_part = min(avg_score, 100) * 0.5
    ielts_part = min(ielts_band, 9) * 5 * 0.3
    profile_part = min(avg_score * 0.2, 20)
    eligibility = round(min(score_part + ielts_part + profile_part, 100))
    dream_unis, match_unis, safety_unis, all_scholarships = [], [], [], []
    for country in countries:
        if country not in UNIVERSITIES:
            continue
        data = UNIVERSITIES[country]
        flag = data["flag"]
        qd = [u for u in data["dream"] if avg_score >= u["min_gpa"] and ielts_band >= u["min_ielts"] and budget_order.get(u["cost"], 2) <= user_budget]
        dream_unis.extend([{"country": country, "flag": flag, "tier": "Dream", **u} for u in qd[:2]])
        qm = [u for u in data["match"] if avg_score >= u["min_gpa"] and ielts_band >= u["min_ielts"] and budget_order.get(u["cost"], 2) <= user_budget]
        match_unis.extend([{"country": country, "flag": flag, "tier": "Match", **u} for u in qm[:2]])
        qs = [u for u in data["safety"] if avg_score >= u["min_gpa"] and ielts_band >= u["min_ielts"]]
        safety_unis.extend([{"country": country, "flag": flag, "tier": "Safety", **u} for u in qs[:2]])
        for s in SCHOLARSHIPS.get(country, []):
            if avg_score >= s["min_score"] and ielts_band >= s["min_ielts"]:
                all_scholarships.append({"country": country, "flag": flag, **s})
    next_steps = []
    if ielts_band < 7.0:
        next_steps.append("📚 Prepare for IELTS — aim for 7.0+ to unlock top universities")
    if avg_score < 85:
        next_steps.append("📈 Raise your percentage to 85+ for Dream-tier universities")
    next_steps.extend([
        "✍️ Write your Statement of Purpose (SOP) — takes 2-3 weeks",
        "📨 Request 2 Letters of Recommendation",
        "💳 Prepare visa documents (bank statements, passport)",
    ])
    if all_scholarships:
        next_steps.append(f"💰 Apply for the {len(all_scholarships)} scholarships you qualify for!")
    return {
        "eligibility_score": eligibility,
        "dream": dream_unis,
        "match": match_unis,
        "safety": safety_unis,
        "scholarships": all_scholarships[:6],
        "next_steps": next_steps[:5],
        "total_universities": len(dream_unis) + len(match_unis) + len(safety_unis)
    }

@app.post("/study_abroad")
async def study_abroad(data: Dict):
    try:
        return check_abroad_eligibility(
            float(data.get("avg_score", 0)),
            float(data.get("ielts_band", 6.5)),
            data.get("countries", []),
            data.get("budget", "medium")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

@app.get("/manifest.json")
async def manifest():
    return FileResponse("manifest.json", media_type="application/json")

@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    if os.path.exists("privacy.html"):
        with open("privacy.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Not found</h1>", status_code=404)

@app.post("/predict")
async def predict(data: StudentInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    try:
        pct = data.percentage
        class_num = 12 if data.level == "Class 12" else 13
        features = np.array([[class_num, data.study_hours, pct, pct, pct]])
        prediction = model.predict(features)[0]
        confidence = float(max(model.predict_proba(features)[0]))

        recommendations = []
        if pct >= 85:
            recommendations.append("🌟 Excellent! Target top universities & competitive exams")
        elif pct >= 75:
            recommendations.append("👍 Good score! Push to 85+ for Dream universities")
        elif pct >= 60:
            recommendations.append("📚 Solid base — focus on weak areas to cross 75%")
        else:
            recommendations.append("💪 Focus on concept building — daily practice will raise your score fast")

        db = SessionLocal()
        try:
            record = StudentRecord(
                student_name=data.student_name,
                board=data.level,
                student_class=data.level,
                stream=data.stream,
                study_hours=data.study_hours,
                avg_score=round(pct, 1),
                subject_scores=json.dumps({"Overall": pct}),
                performance_level=prediction
            )
            db.add(record)
            db.commit()
        finally:
            db.close()

        return {
            "performance_level": prediction,
            "confidence": round(confidence * 100, 1),
            "average_score": round(pct, 1),
            "weak_subjects": [],
            "recommendations": recommendations,
            "careers": suggest_careers_stream(data.stream, data.level, pct)
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