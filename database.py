from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# 1. Database URL Configuration
# Use PostgreSQL on Render, SQLite locally
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    "sqlite:///./students.db"  # Local fallback
)

# Fix for Render's PostgreSQL SSL requirement
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 2. Create Engine & Session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 3. Define Student Record Model
class StudentRecord(Base):
    __tablename__ = "student_records"
    
    id = Column(Integer, primary_key=True, index=True)
    study_hours = Column(Float, nullable=False)
    attendance = Column(Float, nullable=False)
    math_score = Column(Float, nullable=False)
    science_score = Column(Float, nullable=False)
    english_score = Column(Float, nullable=False)
    performance_level = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

# 4. Initialize Database Tables
def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created/verified")

# 5. Dependency for FastAPI (Optional but recommended)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()