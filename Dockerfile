FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# This MUST be present to train the model during build
RUN python train_model.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]