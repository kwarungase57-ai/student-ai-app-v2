FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including train_model.py and any existing model files)
COPY . .

# Train the model during the build process so it exists when the app starts
RUN python train_model.py

# Start the web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]