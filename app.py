"""Industrial-style PlantGuard web interface and audit API."""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from PIL import Image

from src.quality import assess_image
from src.guidance import guidance_for

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "instance" / "plantguard.db"
MODEL_PATH = BASE / "artifacts" / "best_model.pt"
ALLOWED = {"image/jpeg", "image/png"}
MAX_BYTES = 8 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, region TEXT, soil_type TEXT,
        soil_moisture TEXT, crop_hint TEXT, notes TEXT, quality_score REAL NOT NULL,
        quality_json TEXT NOT NULL, prediction TEXT NOT NULL, confidence REAL NOT NULL,
        model_version TEXT NOT NULL, mode TEXT NOT NULL)""")
    return db


def predictor():
    if not MODEL_PATH.exists():
        return None, None, "demo-no-model"
    import torch
    from torchvision import transforms
    from src.model import load_checkpoint
    model, checkpoint = load_checkpoint(MODEL_PATH)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, (transform, checkpoint["classes"]), f"mobilenetv3-epoch-{checkpoint['epoch']}"


MODEL, MODEL_META, MODEL_VERSION = predictor()


@app.get("/")
def index():
    return render_template("index.html", model_ready=MODEL is not None)


@app.post("/api/predict")
def predict():
    upload = request.files.get("image")
    if not upload or upload.mimetype not in ALLOWED:
        return jsonify(error="Upload a JPEG or PNG image."), 400
    payload = upload.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        return jsonify(error="Image exceeds the 8 MB limit."), 413
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.verify() if False else None
    except Exception:
        return jsonify(error="The uploaded file is not a valid image."), 400
    quality = assess_image(image)
    if MODEL is None:
        label, confidence, mode = "Model not trained", 0.0, "demo"
    else:
        import torch
        transform, classes = MODEL_META
        with torch.no_grad():
            probabilities = MODEL(transform(image).unsqueeze(0)).softmax(1)[0]
        confidence, idx = probabilities.max(0)
        label, confidence, mode = classes[idx.item()], float(confidence), "trained"
    record_id = str(uuid.uuid4())
    created = datetime.now(timezone.utc).isoformat()
    values = (
        record_id, created, request.form.get("region", "Unknown"),
        request.form.get("soil_type", "Unknown"), request.form.get("soil_moisture", "Unknown"),
        request.form.get("crop_hint", "Unknown"), request.form.get("notes", "")[:500],
        quality.score, json.dumps(quality.to_dict()), label, confidence, MODEL_VERSION, mode,
    )
    with get_db() as db:
        db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
    return jsonify(id=record_id, timestamp=created, prediction=label,
                   confidence=round(confidence, 4), mode=mode, quality=quality.to_dict(),
                   guidance=guidance_for(label),
                   disclaimer="Research use only; consult a qualified agricultural expert.")


@app.get("/api/history")
def history():
    with get_db() as db:
        rows = db.execute("""SELECT id,created_at,region,soil_type,soil_moisture,crop_hint,
                            quality_score,prediction,confidence,model_version,mode
                            FROM predictions ORDER BY created_at DESC LIMIT 100""").fetchall()
    return jsonify([dict(row) for row in rows])


@app.get("/api/export.csv")
def export_csv():
    with get_db() as db:
        rows = db.execute("SELECT * FROM predictions ORDER BY created_at").fetchall()
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(dict(row) for row in rows)
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=prediction_log.csv"})


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_ready=MODEL is not None, model_version=MODEL_VERSION)


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
