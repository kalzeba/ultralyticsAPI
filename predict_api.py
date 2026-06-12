from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Chargement unique du modèle au démarrage
# yolov8n = le plus léger, idéal CPU (3MB vs 22MB pour yolov8s)
model = YOLO("yolov8n.pt")
CLASS_NAMES = model.names

MAX_SIZE = 640  # taille max avant resize


def decode_image(b64: str):
    """Décode une image base64 (avec ou sans header data:image/...)"""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def resize_if_needed(img):
    """Redimensionne uniquement si l'image dépasse MAX_SIZE (évite calcul inutile)"""
    h, w = img.shape[:2]
    if max(h, w) <= MAX_SIZE:
        return img
    scale = MAX_SIZE / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "yolov8n",
        "classes": len(CLASS_NAMES)
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "missing 'image' field"}), 400

    img = decode_image(data["image"])
    if img is None:
        return jsonify({"error": "image decode failed"}), 400

    img = resize_if_needed(img)

    results = model.predict(
        img,
        imgsz=320,   # inférence rapide sur CPU
        conf=0.25,
        iou=0.45,
        verbose=False
    )[0]

    h_img, w_img = img.shape[:2]
    detections = []

    for box in results.boxes:
        x, y, bw, bh = box.xywh[0].tolist()
        cls_id = int(box.cls[0])
        detections.append({
            "class": CLASS_NAMES.get(cls_id, f"object_{cls_id}"),
            "class_id": cls_id,
            "confidence": round(float(box.conf[0]), 4),
            "bbox": {
                "x": round((x - bw / 2) / w_img * 100, 2),
                "y": round((y - bh / 2) / h_img * 100, 2),
                "w": round(bw / w_img * 100, 2),
                "h": round(bh / h_img * 100, 2),
            },
        })

    return jsonify({
        "count": len(detections),
        "items": detections
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
