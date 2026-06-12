# Stocko — YOLO Detection API

API Flask légère pour la détection d'objets sur étagère, utilisée par l'app **Stocko**.  
Optimisée pour déploiement CPU sur [Render](https://render.com).

## Routes

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Détection sur une image |

## Format de la requête `/predict`

```json
{
  "image": "<base64 avec ou sans header data:image/...>"
}
```

## Format de la réponse

```json
{
  "count": 3,
  "items": [
    {
      "class": "bottle",
      "class_id": 39,
      "confidence": 0.872,
      "bbox": { "x": 12.5, "y": 8.3, "w": 10.2, "h": 25.6 }
    }
  ]
}
```

Les coordonnées `bbox` sont en **pourcentage** (0–100) de la taille de l'image.

## Déploiement sur Render

1. Connecter ce repo sur [render.com](https://render.com)
2. Type : **Web Service**, Environment : **Docker**
3. Le build télécharge automatiquement `yolov8n.pt`

## Développement local

```bash
pip install -r requirements.txt
python predict_api.py
# → http://localhost:5000
```
