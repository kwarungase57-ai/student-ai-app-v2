import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n = 1500
data = {
    'student_class': np.random.choice([6,7,8,9,10,11,12], n),
    'study_hours': np.random.randint(5, 40, n),
    'avg_score': np.random.randint(15, 100, n),
    'min_score': np.random.randint(10, 90, n),
    'max_score': np.random.randint(40, 100, n)
}
df = pd.DataFrame(data)
df['min_score'] = df[['min_score','avg_score']].min(axis=1)
df['max_score'] = df[['max_score','avg_score']].max(axis=1)

def get_perf(row):
    if row['avg_score'] >= 80: return 'Excellent'
    elif row['avg_score'] >= 60: return 'Good'
    elif row['avg_score'] >= 40: return 'Average'
    else: return 'Needs Improvement'

df['performance_level'] = df.apply(get_perf, axis=1)

features = ['student_class', 'study_hours', 'avg_score', 'min_score', 'max_score']
X = df[features]
y = df['performance_level']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'student_model.pkl')
print(f"✅ Subject-independent model saved with {len(features)} features!")