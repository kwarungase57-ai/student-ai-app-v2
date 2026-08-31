import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('generated_dataset.csv')  # Make sure this exists

def get_perf(row):
    avg = (row['math_score'] + row['science_score'] + row['english_score']) / 3
    if avg >= 80: return 'Excellent'
    elif avg >= 60: return 'Good'
    elif avg >= 40: return 'Average'
    else: return 'Needs Improvement'

df['performance_level'] = df.apply(get_perf, axis=1)

features = ['student_class', 'study_hours', 'attendance', 'math_score', 'science_score', 'english_score']
X = df[features]
y = df['performance_level']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'student_model.pkl')
print(f"✅ Model saved with {len(features)} features!")