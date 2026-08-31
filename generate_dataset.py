import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    'student_class': np.random.randint(6, 13, n_samples),
    'study_hours': np.random.uniform(5, 30, n_samples),
    'attendance': np.random.uniform(50, 100, n_samples),
    'math_score': np.random.uniform(30, 100, n_samples),
    'science_score': np.random.uniform(30, 100, n_samples),
    'english_score': np.random.uniform(30, 100, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('generated_dataset.csv', index=False)
print("✅ Dataset generated: generated_dataset.csv")