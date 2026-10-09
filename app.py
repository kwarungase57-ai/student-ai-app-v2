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

app = FastAPI(title="AI Student Analyzer & Study Abroad Platform")
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
    student_name: str = "Student"
    board: str
    student_class: str
    stream: str = "None"
    study_hours: float
    start_time: str = "16:00"
    subject_scores: Dict[str, float]

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

class AbroadInput(BaseModel):
    student_name: str = "Student"
    avg_score: float
    ielts_band: float = 6.5
    countries: List[str] = ["UK", "USA", "Canada", "Australia", "Germany"]
    budget: str = "medium"  # low / medium / high

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
                    return f"📐 Calculation: {part} = {round(val, 4)}\n\nTip: Follow BODMAS order."
                except Exception:
                    continue
    ql = q.lower()
    for key, ans in CANNED.items():
        if key in ql:
            return ans
    return "🤔 Try asking: 'What is photosynthesis?', 'Solve 12*8+4', or any study question!"

@app.post("/ask")
async def ask(data: DoubtInput):
    context = (f"Student profile: {data.board} {data.student_class}, stream: {data.stream}, "
               f"weak subjects: {', '.join(data.weak_subjects) if data.weak_subjects else 'none'}.")
    ql = data.question.lower()
    wants_chapter = any(w in ql for w in ["chapter", "notes", "summary", "text of", "full topic"])
    if wants_chapter:
        prompt = (f"You are a friendly personal tutor. {context} Provide a study summary with: "
                  f"1) Chapter overview 2) Key concepts 3) Formulas 4) One solved example 5) Revision tips. Under 500 words.\n\n"
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
        {"q": "Past tense of 'Go'?", "a": ["Goed","Went","Gone","Going"], "c": 1},
    ],
    "Social": [
        {"q": "First PM of India?", "a": ["Nehru","Gandhi","Patel","Rajendra Prasad"], "c": 0},
        {"q": "Longest river in India?", "a": ["Yamuna","Ganga","Godavari","Brahmaputra"], "c": 1},
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
        prompt = (f"You are an expert teacher for {data.board} {data.student_class}. "
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

# ---------- CAREER GUIDANCE ----------
CAREER_TRACKS = {
    "Engineering & Technology": ["Physics", "Math"],
    "Software, IT & AI": ["Computer Science", "Computer Applications", "Math"],
    "Medical & Healthcare": ["Biology", "Chemistry", "Science"],
    "Pure Science & Research": ["Physics", "Chemistry", "Science"],
    "Data Science & Statistics": ["Math", "Economics"],
    "Commerce, Finance & CA": ["Accountancy", "Business Studies", "Economics"],
    "Business & Management": ["Business Studies", "Economics", "English"],
    "Law, Civil Services & Administration": ["Political Science", "History", "Social Studies"],
    "Humanities, Teaching & Psychology": ["History", "Geography", "Hindi"],
    "Media, Writing & Languages": ["English", "Hindi", "Marathi"],
    "Design, Arts & Creativity": ["English", "Social Studies", "Science"],
}

CAREER_ROLES = {
    "Engineering & Technology": ["Mechanical / Civil Engineer", "Robotics Specialist", "ISRO / DRDO Scientist"],
    "Software, IT & AI": ["Software Developer", "AI / ML Engineer", "Cybersecurity Expert"],
    "Medical & Healthcare": ["Doctor (MBBS)", "Pharmacist", "Biotech Scientist"],
    "Pure Science & Research": ["Research Scientist", "Astrophysicist", "Lab Specialist"],
    "Data Science & Statistics": ["Data Scientist", "Statistician", "Business Analyst"],
    "Commerce, Finance & CA": ["Chartered Accountant", "Investment Banker", "Financial Advisor"],
    "Business & Management": ["Entrepreneur", "Marketing Manager", "HR Manager"],
    "Law, Civil Services & Administration": ["IAS / IPS Officer", "Lawyer / Judge", "Policy Analyst"],
    "Humanities, Teaching & Psychology": ["Teacher / Professor", "Psychologist", "NGO Leader"],
    "Media, Writing & Languages": ["Journalist", "Content Writer", "Translator"],
    "Design, Arts & Creativity": ["Graphic / UI Designer", "Animator", "Architect"],
}

def suggest_careers(subject_scores):
    results = []
    for track, subjects in CAREER_TRACKS.items():
        present = [s for s in subjects if s in subject_scores]
        if not present:
            continue
        avg = sum(subject_scores[s] for s in present) / len(present)
        results.append({"track": track, "roles": CAREER_ROLES[track], "match": round(avg)})
    results.sort(key=lambda x: x["match"], reverse=True)
    return results[:3]

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
        "flag": "🇺🇸",
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
    
    # Overall eligibility score (weighted)
    score_part = min(avg_score, 100) * 0.5
    ielts_part = min(ielts_band, 9) * 5 * 0.3
    profile_part = min(avg_score * 0.2, 20)
    eligibility = round(min(score_part + ielts_part + profile_part, 100))
    
    # University matches by tier
    dream_unis, match_unis, safety_unis = [], [], []
    all_scholarships = []
    
    for country in countries:
        if country not in UNIVERSITIES:
            continue
        data = UNIVERSITIES[country]
        flag = data["flag"]
        
        # Dream: top 2 that qualify
        qualified_dream = [u for u in data["dream"]
                           if avg_score >= u["min_gpa"] and ielts_band >= u["min_ielts"]
                           and budget_order.get(u["cost"], 2) <= user_budget]
        dream_unis.extend([{"country": country, "flag": flag, "tier": "Dream", **u} for u in qualified_dream[:2]])
        
        # Match: top 2
        qualified_match = [u for u in data["match"]
                           if avg_score >= u["min_gpa"] and ielts_band >= u["min_ielts"]
                           and budget_order.get(u["cost"], 2) <= user_budget]
        match_unis.extend([{"country": country, "flag": flag, "tier": "Match", **u} for u in qualified_match[:2]])
        
        # Safety: top 2
        qualified_safety = [u for u in data["safety"]
                            if avg_score >= u["min_gpa"] and ielts_band >= u["min_ielts"]]
        safety_unis.extend([{"country": country, "flag": flag, "tier": "Safety", **u} for u in qualified_safety[:2]])
        
        # Scholarships
        for s in SCHOLARSHIPS.get(country, []):
            if avg_score >= s["min_score"] and ielts_band >= s["min_ielts"]:
                all_scholarships.append({"country": country, "flag": flag, **s})
    
    # Next steps based on profile
    next_steps = []
    if ielts_band < 7.0:
        next_steps.append("📚 Prepare for IELTS — aim for 7.0+ to unlock top universities")
    if avg_score < 85:
        next_steps.append("📈 Focus on raising your GPA to 85+ for Dream-tier universities")
    if not dream_unis:
        next_steps.append("🎯 Your profile qualifies for Match universities — strengthen it for Dream schools")
    next_steps.extend([
        "✍️ Start writing your Statement of Purpose (SOP) — takes 2-3 weeks",
        "📨 Request 2 Letters of Recommendation from teachers",
        "💳 Prepare for visa documentation (bank statements, passport)",
    ])
    if all_scholarships:
        next_steps.append(f"💰 Apply for {len(all_scholarships)} scholarships you qualify for!")
    
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
async def study_abroad(data: AbroadInput):
    try:
        result = check_abroad_eligibility(
            data.avg_score, data.ielts_band, data.countries, data.budget
        )
        return result
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

def get_advice(subject, score):
    e = emoji_for(subject)
    if score < 40:
        return f"{e} {subject} Critical: start from basics. Watch video lectures daily + solve NCERT examples"
    if score < 60:
        return f"{e} {subject} Weak: practice 30–45 mins daily, revise notes and solve exercises"
    if score < 75:
        return f"{e} {subject} Moderate: solve previous year papers and focus on tricky topics"
    return None

def add_minutes(h, m, mins):
    total = h * 60 + m + int(mins)
    return (total // 60) % 24, total % 60

def fmt_time(h, m):
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0: h12 = 12
    return f"{h12}:{m:02d} {suffix}"

def activity_for(score):
    if score < 40: return "Concept building: video lecture + NCERT reading"
    if score < 60: return "Solved examples + exercise questions"
    if score < 75: return "Practice set + previous year questions"
    return "Advanced questions + speed revision"

def generate_timetable(study_hours, subject_scores, start_time="16:00"):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    total_gap = sum(max(0, 100 - s) for s in subject_scores.values())
    if total_gap == 0:
        weights = {s: 1/len(subject_scores) for s in subject_scores}
    else:
        weights = {s: max(0, 100 - sc) / total_gap for s, sc in subject_scores.items()}
    ordered = sorted(subject_scores.keys(), key=lambda s: subject_scores[s])
    weakest = ordered[0]
    try:
        sh, sm = [int(x) for x in start_time.split(":")]
    except Exception:
        sh, sm = 16, 0
    daily_minutes = (study_hours * 60) / 6.0
    timetable = []
    for i, day in enumerate(days):
        tasks = []
        h, m = sh, sm
        if day == "Sunday":
            tasks.append(f"📝 {fmt_time(h, m)} – Weekly Mock Test: {weakest} (60 min)")
            h, m = add_minutes(h, m, 70)
            tasks.append(f"🔁 {fmt_time(h, m)} – Review mistakes (30 min)")
            tasks.append(f"🧘 {fmt_time(h, m)} – Rest & light reading")
        else:
            order = ordered[i % len(ordered):] + ordered[:i % len(ordered)]
            for idx, subj in enumerate(order):
                mins = int(round(daily_minutes * weights[subj] / 5) * 5)
                if mins < 20: continue
                eh, em = add_minutes(h, m, mins)
                tasks.append(f"{emoji_for(subj)} {fmt_time(h, m)}–{fmt_time(eh, em)} {subj}: {activity_for(subject_scores[subj])} ({mins} min)")
                if idx < len(order) - 1:
                    tasks.append(f"☕ {fmt_time(eh, em)} – Break (10 min)")
                h, m = add_minutes(eh, em, 10)
            tasks.append("✏️ Homework / Assignments")
            tasks.append(f"🎯 Night: revise {weakest} for 10 min")
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
            if advice: recommendations.append(advice)
        if data.study_hours < 14:
            recommendations.append("⏰ Low study hours! Aim for 2-3 hours/day")
        if not weak_subjects:
            recommendations.append("🌟 Excellent scores! Focus on advanced problems")
        timetable = generate_timetable(data.study_hours, data.subject_scores, data.start_time)
        db = SessionLocal()
        try:
            record = StudentRecord(
                student_name=data.student_name,
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
            "timetable": timetable,
            "careers": suggest_careers(data.subject_scores)
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

@app.get("/records")
async def get_records():
    db = SessionLocal()
    try:
        records = db.query(StudentRecord).order_by(StudentRecord.timestamp.desc()).limit(100).all()
        return [{
            "timestamp": r.timestamp.isoformat(),
            "student_name": r.student_name,
            "board": r.board,
            "student_class": r.student_class,
            "stream": r.stream,
            "study_hours": r.study_hours,
            "avg_score": r.avg_score,
            "subject_scores": json.loads(r.subject_scores),
            "performance_level": r.performance_level
        } for r in records]
    finally:
        db.close()