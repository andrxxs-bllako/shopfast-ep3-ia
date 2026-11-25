"""
Script de inferencia estándar para AWS SageMaker SKLearn Container (IE4).
Implementa model_fn, input_fn, predict_fn y output_fn para servir
el modelo shopfast_recommender.joblib en un Endpoint administrado de SageMaker.
"""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np


def model_fn(model_dir: str):
    """Carga el modelo serializado desde /opt/ml/model en SageMaker."""
    model_path = os.path.join(model_dir, "shopfast_recommender.joblib")
    return joblib.load(model_path)


def input_fn(request_body: str | bytes, request_content_type: str = "application/json") -> np.ndarray:
    """Deserializa el payload JSON enviado por SageMaker Runtime."""
    if request_content_type != "application/json":
        raise ValueError(f"Content-Type no soportado: {request_content_type}")
    payload = json.loads(request_body)
    instances = payload.get("instances", payload)
    return np.array(instances, dtype=float)


def predict_fn(input_data: np.ndarray, model: Any) -> list[float]:
    """Ejecuta inferencia de probabilidad de compra en CPU."""
    probs = model.predict_proba(input_data)[:, 1]
    return [round(float(p), 5) for p in probs]


def output_fn(prediction: list[float], accept: str = "application/json") -> tuple[str, str]:
    """Serializa la respuesta JSON estándar para el cliente FastAPI."""
    return json.dumps({"predictions": prediction}), "application/json"
