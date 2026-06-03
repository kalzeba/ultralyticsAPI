from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# yolov8n : déjà dans le repo (6.5MB), pas de téléchargement au démarrage
model = YOLO("yolov8n.pt")
model.fuse()
CLASS_NAMES = model.names

def decode_image(b64: str):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "yolov8n", "classes": len(CLASS_NAMES)})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "missing image"}), 400
        img = decode_image(data["image"])
        if img is None:
            return jsonify({"error": "invalid image"}), 400
        h, w = img.shape[:2]
        img = preprocess(img)
        results = model(img, conf=0.35, iou=0.45, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x, y, bw, bh = box.xywh[0].tolist()
            cls_id = int(box.cls[0])
            detections.append({
                "class": CLASS_NAMES.get(cls_id, f"object_{cls_id}"),
                "class_id": cls_id,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x": round((x - bw / 2) / w * 100, 2),
                    "y": round((y - bh / 2) / h * 100, 2),
                    "w": round(bw / w * 100, 2),
                    "h": round(bh / h * 100, 2),
                },
            })
        return jsonify({"items": detections, "count": len(detections)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
