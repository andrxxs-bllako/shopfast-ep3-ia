"""Configuración centralizada del microservicio ShopFast vía variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Parámetros operativos, de seguridad, resiliencia y mensajería."""

    api_title: str = "ShopFast AI — API de Recomendación Empresarial"
    api_version: str = "2.0.0"
    api_key: str = os.getenv("SHOPFAST_API_KEY", "shopfast-ep3-key-2026")
    anonymization_salt: str = os.getenv("ANONYMIZATION_SALT", "shopfast-ley19628-salt")
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "120"))
    queue_backend: str = os.getenv("QUEUE_BACKEND", "local").lower()
    sqs_queue_url: str = os.getenv("SQS_QUEUE_URL", "")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    rabbitmq_queue: str = os.getenv("RABBITMQ_QUEUE", "shopfast_events")
    sagemaker_endpoint_name: str = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    circuit_breaker_timeout_ms: float = float(os.getenv("CIRCUIT_BREAKER_TIMEOUT_MS", "180"))
    model_path: Path = BASE_DIR / "models" / "shopfast_recommender.joblib"
    metadata_path: Path = BASE_DIR / "models" / "model_metadata.json"
    catalog_path: Path = BASE_DIR / "models" / "product_catalog.json"
    local_queue_path: Path = BASE_DIR / "data" / "local_queue.jsonl"
    processed_telemetry_path: Path = BASE_DIR / "data" / "processed_telemetry.json"


def get_settings() -> Settings:
    """Retorna configuración actualizada desde entorno."""
    return Settings()
