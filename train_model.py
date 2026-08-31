import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n = 1500
data = {
    'student_class': np.random.choice([6,7,8,9,10,11,12], n),
    'career_stream': np.random.choice([1,2,3,4], n),
    'study_hours': np.random.randint(5, 40, n),
    'math_score': np.random.randint(15, 100, n),
    'science_score': np.random.randint(15, 100, n),
    'english_score': np.random.randint(15, 100, n)
}
df = pd.DataFrame(data)

def get_perf(row):
    avg = (row['math_score'] + row['science_score'] + row['english_score']) / 3
    if avg >= 80: return 'Excellent'
    elif avg >= 60: return 'Good'
    elif avg >= 40: return 'Average'
    else: return 'Needs Improvement'

df['performance_level'] = df.apply(get_perf, axis=1)

features = ['student_class', 'career_stream', 'study_hours', 'math_score', 'science_score', 'english_score']
X = df[features]
y = df['performance_level']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'student_model.pkl')
print(f"✅ Model saved with {len(features)} features!")