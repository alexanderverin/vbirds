"""Этап 8 — FastAPI /predict (обёртка над CV-модулем овоскопирования).

Принимает фото яйца → возвращает класс + confidence + честную пометку об ограничении.
Backend лёгкий: baseline-модель без torch (держим образ лёгким).
Порт фиксированный 8000.

Запуск: uvicorn main:app --host 0.0.0.0 --port 8000   (из папки backend/)
"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from model import BAD_PHOTO_MSG, DISCLAIMER, looks_like_candling, predict_pil

app = FastAPI(title="VerinBirds — Candling API", version="0.1.0-MVP")


@app.get("/health")
def health():
    return {"status": "ok", "model": "baseline_svm", "stage": "MVP", "trained_on": "chicken"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    ok, why = looks_like_candling(pil)
    if not ok:
        raise HTTPException(status_code=422, detail=f"{BAD_PHOTO_MSG} (причина: {why})")

    top, probs = predict_pil(pil)
    return {
        "prediction": top,
        "confidence": round(probs[top], 4),
        "probabilities": {k: round(v, 4) for k, v in sorted(probs.items(), key=lambda x: -x[1])},
        "model": "baseline_svm (цветовые гистограммы + GLCM-текстура)",
        "trained_on": "chicken (hlnkmb/chicken-egg-analysis-dataset)",
        "validated_on_quail": False,
        "disclaimer": DISCLAIMER,
    }


@app.get("/")
def root():
    return {"service": "VerinBirds candling API (MVP)",
            "endpoints": {"POST /predict": "фото -> класс+confidence", "GET /health": "статус"},
            "disclaimer": DISCLAIMER}
