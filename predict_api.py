from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import base64

app = Flask(__name__)

model = YOLO("yolov8n.pt")
model.fuse()

def decode_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "missing image"}), 400

        img = decode_image(data["image"])

        if img is None:
            return jsonify({"error": "invalid image"}), 400

        results = model(img)[0]

        h, w = img.shape[:2]

        detections = []

        for box in results.boxes:
            x, y, bw, bh = box.xywh[0].tolist()

            detections.append({
                "class": int(box.cls[0]),
                "confidence": round(float(box.conf[0]), 4),

                # conversion en %
                "bbox": {
                    "x": round((x - bw/2) / w * 100, 2),
                    "y": round((y - bh/2) / h * 100, 2),
                    "w": round(bw / w * 100, 2),
                    "h": round(bh / h * 100, 2)
                }
            })

        return jsonify({
            "items": detections,
            "count": len(detections)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}
