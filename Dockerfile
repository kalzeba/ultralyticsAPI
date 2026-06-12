# Image Python slim — ~200MB au lieu de ~8GB avec ultralytics/ultralytics
FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales pour OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharger le modèle au build (évite le download à la 1ère requête)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY predict_api.py .

EXPOSE 5000

# 1 worker (CPU), timeout 120s pour les grosses images
CMD ["gunicorn", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:5000", "predict_api:app"]
