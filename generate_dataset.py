import pandas as pd
import numpy as np

print("Generating realistic student dataset...")

np.random.seed(42)
n_students = 2000

# Create correlated features (real students who study more usually have better attendance)
study_hours = np.random.normal(10, 4, n_students).clip(0, 30)
attendance = np.random.normal(75 + (study_hours * 1.5), 12, n_students).clip(0, 100)
prev_score = np.random.normal(60 + (study_hours * 2), 15, n_students).clip(0, 100)
assignments = np.random.poisson(5 + (study_hours * 0.3), n_students).clip(0, 10)

# Add some realistic noise and outliers
noise = np.random.normal(0, 8, n_students)
final_score = (study_hours * 1.8) + (attendance * 0.4) + (prev_score * 0.5) + (assignments * 2.0) + noise
final_score = final_score.clip(0, 100)

# Create DataFrame
df = pd.DataFrame({
    'study_hours': np.round(study_hours, 1),
    'attendance_pct': np.round(attendance, 1),
    'previous_test_score': np.round(prev_score, 1),
    'assignments_completed': assignments.astype(int),
    'final_exam_score': np.round(final_score, 1)
})

# Save to CSV
df.to_csv('student_data.csv', index=False)
print(f"✅ Dataset saved! {len(df)} students with {len(df.columns)} features.")
print(df.head())