"""Microservicio API REST de ShopFast (EP3) con FastAPI, OpenAPI 3.1, Seguridad, ML y SQS/RabbitMQ."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.messaging import EventPublisher
from app.model_service import RecommenderService
from app.security import anonymize_user_id, verify_api_key

settings = get_settings()
service = RecommenderService()
publisher = EventPublisher()


class TelemetryTracker:
    """Almacena métricas RED en tiempo real (Rate, Errors, Duration)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_errors = 0
        self.latencies_ms: list[float] = []

    def record(self, duration_ms: float, status_code: int) -> None:
        with self._lock:
            self.total_requests += 1
            if status_code >= 500:
                self.total_errors += 1
            self.latencies_ms.append(duration_ms)
            if len(self.latencies_ms) > 5000:
                self.latencies_ms = self.latencies_ms[-5000:]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            arr = np.array(self.latencies_ms, dtype=float) if self.latencies_ms else np.array([0.0])
            err_rate = (self.total_errors / self.total_requests) * 100.0 if self.total_requests else 0.0
            return {
                "total_requests": self.total_requests,
                "total_5xx_errors": self.total_errors,
                "error_rate_pct": round(err_rate, 3),
                "latency_ms": {
                    "p50": round(float(np.percentile(arr, 50)), 3),
                    "p95": round(float(np.percentile(arr, 95)), 3),
                    "p99": round(float(np.percentile(arr, 99)), 3),
                    "mean": round(float(np.mean(arr)), 3),
                },
            }


telemetry = TelemetryTracker()

app = FastAPI(
    title=settings.api_title,
    description=(
        "API REST empresarial para recomendaciones personalizadas de ShopFast (EP3 — ITY1102). "
        "Incluye modelo ML real GradientBoosting, interpretabilidad para toma de decisiones, "
        "seguridad (X-API-Key, Rate Limiting, cabeceras TLS, Ley N° 19.628), Circuit Breaker "
        "y publicación asíncrona hacia Amazon SQS / RabbitMQ."
    ),
    version=settings.api_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_and_telemetry_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    if not request.url.path.startswith(("/docs", "/openapi.json", "/redoc")):
        telemetry.record(elapsed_ms, response.status_code)

    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"

    if hasattr(request.state, "rate_limit"):
        response.headers["X-RateLimit-Limit"] = str(request.state.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(request.state.rate_remaining)
    return response


class DatosEntrada(BaseModel):
    """Esquema de entrada para solicitar recomendaciones personalizadas en ShopFast."""

    user_id: str = Field(default="cliente_1045", min_length=2, max_length=64, examples=["cliente_1045"])
    categoria_preferida: str = Field(default="Laptops", examples=["Laptops"])
    presupuesto_usd: float = Field(default=950.0, gt=10.0, le=10000.0, examples=[950.0])
    vistas_previas_categoria: int = Field(default=5, ge=0, le=100, examples=[5])
    dias_desde_ultima_compra: int = Field(default=14, ge=0, le=3650, examples=[14])
    items_en_carrito: int = Field(default=1, ge=0, le=50, examples=[1])
    es_cyberday: bool = Field(default=False, examples=[False])
    top_k: int = Field(default=5, ge=1, le=10, examples=[5])
    simular_fallo: bool = Field(default=False, description="Activa manualmente el Circuit Breaker para demostración en vivo.")


class ClickstreamEvent(BaseModel):
    """Evento de navegación del usuario en ShopFast para ingesta asíncrona."""

    user_id: str = Field(..., min_length=2, max_length=64, examples=["cliente_1045"])
    producto_id: str = Field(..., min_length=3, max_length=32, examples=["SF-LAP-0001"])
    accion: str = Field(..., pattern="^(view|click|add_to_cart|purchase)$", examples=["add_to_cart"])
    precio_usd: float = Field(..., gt=0.0, examples=[899.99])


@app.get("/", tags=["Sistema"])
def home() -> dict[str, Any]:
    """Endpoint raíz de verificación rápida del servicio."""
    return {
        "servicio": settings.api_title,
        "version": settings.api_version,
        "estado": "operativo",
        "modelo": service.metadata["model_name"],
        "documentacion": "/docs",
    }


@app.get("/health", tags=["Sistema"])
def health_check() -> dict[str, Any]:
    """Health check utilizado por AWS Application Load Balancer (Target Group)."""
    return {
        "status": "healthy",
        "model_loaded": service.model is not None,
        "catalog_items": len(service.catalog),
        "circuit_breaker": service.circuit_state,
        "queue_backend": settings.queue_backend,
    }


@app.get("/model/info", tags=["Modelo IA"])
def model_info() -> dict[str, Any]:
    """Retorna metadatos, hiperparámetros, métricas reales e importancia de variables del modelo."""
    return service.metadata


@app.post("/predecir", tags=["Recomendaciones IA"])
def predecir(
    datos: DatosEntrada,
    request: Request,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Genera Top-K recomendaciones personalizadas con interpretabilidad y publica evento asíncrono."""
    anon_user = anonymize_user_id(datos.user_id)
    req_dict = datos.model_dump()
    result = service.recommend(req_dict)

    top_product = result["recomendaciones"][0]["producto_id"] if result["recomendaciones"] else None
    event_status = publisher.publish(
        "recommendation_served",
        {
            "user_hash": anon_user,
            "categoria_preferida": datos.categoria_preferida,
            "origen_inferencia": result["origen_inferencia"],
            "top_producto_id": top_product,
            "latencia_ms": result["latencia_inferencia_ms"],
        },
    )

    return {
        "estado": "éxito",
        "user_hash_anonimizado": anon_user,
        "cumplimiento_privacidad": "Ley N° 19.628 (SHA-256 sin PII)",
        "origen_inferencia": result["origen_inferencia"],
        "cache_hit": result["cache_hit"],
        "circuit_breaker_activo": result["circuit_breaker_activo"],
        "latencia_inferencia_ms": result["latencia_inferencia_ms"],
        "evento_asincrono": event_status,
        "recomendaciones": result["recomendaciones"],
        "correlation_id": request.headers.get("X-Correlation-ID", "auto"),
    }


@app.get("/recommendations/{user_id}", tags=["Recomendaciones IA"])
def get_recommendations_by_user(
    user_id: str,
    request: Request,
    categoria: str = Query(default="Laptops"),
    presupuesto_usd: float = Query(default=900.0, gt=10.0),
    top_k: int = Query(default=5, ge=1, le=10),
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Endpoint RESTful por identificador de cliente alineado al diagrama C4 (EP2)."""
    datos = DatosEntrada(
        user_id=user_id,
        categoria_preferida=categoria,
        presupuesto_usd=presupuesto_usd,
        top_k=top_k,
    )
    return predecir(datos=datos, request=request, _api_key=_api_key)


@app.post("/events/clickstream", tags=["Mensajería Distribuida"])
def ingest_clickstream(
    evento: ClickstreamEvent,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Publica interacciones de navegación hacia Amazon SQS / RabbitMQ sin bloquear al cliente."""
    anon_user = anonymize_user_id(evento.user_id)
    res = publisher.publish(
        "clickstream_interaction",
        {
            "user_hash": anon_user,
            "producto_id": evento.producto_id,
            "accion": evento.accion,
            "precio_usd": evento.precio_usd,
        },
    )
    return {
        "estado": "encolado",
        "user_hash_anonimizado": anon_user,
        "mensajeria": res,
    }


@app.get("/metrics", tags=["Observabilidad"])
def get_metrics() -> dict[str, Any]:
    """Expone métricas operacionales RED, eficiencia de caché, Circuit Breaker y mensajería."""
    total_cache = service.cache_hits + service.cache_misses
    hit_rate = (service.cache_hits / total_cache) * 100.0 if total_cache else 0.0
    return {
        "red_metrics": telemetry.summary(),
        "cache": {
            "hits": service.cache_hits,
            "misses": service.cache_misses,
            "hit_rate_pct": round(hit_rate, 2),
        },
        "resiliencia": {
            "circuit_breaker_state": service.circuit_state,
            "fallback_activations": service.fallback_count,
            "sagemaker_invocations": service.sagemaker_calls,
        },
        "mensajeria": {
            "configured_backend": publisher.backend,
            "last_backend_used": publisher.last_backend_used,
            "events_published": publisher.published_count,
        },
    }
