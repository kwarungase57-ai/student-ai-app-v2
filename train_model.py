import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

print("Starting training...")

np.random.seed(42)
n = 1000

study_hours = np.random.randint(1, 20, n)
attendance = np.random.randint(40, 100, n)
prev_score = np.random.randint(30, 95, n)
assignments = np.random.randint(0, 10, n)

final_score = (study_hours * 1.8) + (attendance * 0.4) + (prev_score * 0.6) + (assignments * 2.5) + np.random.normal(0, 4, n)
final_score = np.clip(final_score, 0, 100)

X = np.column_stack((study_hours, attendance, prev_score, assignments))
y = final_score

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'student_model.pkl')

print("Success! Model saved as student_model.pkl")
