"""
Suite de 12 pruebas unitarias, de contrato, seguridad, resiliencia e interoperabilidad
para la API y modelo ML de ShopFast (EP3 — IE10).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app, service
from app.messaging import EventConsumer
from app.security import rate_limiter

client = TestClient(app)
VALID_KEY = get_settings().api_key
HEADERS = {"X-API-Key": VALID_KEY}


def setup_function() -> None:
    rate_limiter.set_limit(1000)
    rate_limiter.reset()


def test_01_model_artifacts_and_metrics_quality() -> None:
    """TC-01 (Unitaria): Verifica existencia del modelo serializado y métricas reales >= 0.80."""
    settings = get_settings()
    assert settings.model_path.exists()
    assert settings.metadata_path.exists()
    assert settings.catalog_path.exists()

    meta = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    metrics = meta["metrics"]
    for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert 0.80 <= metrics[name] <= 0.99, f"Métrica {name} fuera de rango realista: {metrics[name]}"


def test_02_unit_inference_latency_sla() -> None:
    """TC-02 (Unitaria): Inferencia directa del modelo cumple SLA < 100 ms y ordena scores."""
    service.clear_cache()
    res = service.recommend(
        {
            "user_id": "unit_user_1",
            "categoria_preferida": "Laptops",
            "presupuesto_usd": 1100.0,
            "vistas_previas_categoria": 7,
            "dias_desde_ultima_compra": 10,
            "items_en_carrito": 1,
            "es_cyberday": True,
            "top_k": 5,
            "simular_fallo": False,
        }
    )
    assert len(res["recomendaciones"]) == 5
    assert res["latencia_inferencia_ms"] < 100.0
    scores = [r["score_afinidad"] for r in res["recomendaciones"]]
    assert scores == sorted(scores, reverse=True)


def test_03_health_and_openapi_contract() -> None:
    """TC-03 (Contrato): /health y esquema OpenAPI exponen todos los endpoints oficiales."""
    r_health = client.get("/health")
    assert r_health.status_code == 200
    body = r_health.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["catalog_items"] >= 200

    r_oas = client.get("/openapi.json")
    assert r_oas.status_code == 200
    paths = r_oas.json()["paths"]
    for ep in ("/health", "/model/info", "/predecir", "/recommendations/{user_id}", "/events/clickstream", "/metrics"):
        assert ep in paths


def test_04_security_missing_api_key_401() -> None:
    """TC-04 (Seguridad): Petición sin cabecera X-API-Key debe ser rechazada con HTTP 401."""
    r = client.post("/predecir", json={"user_id": "u1", "categoria_preferida": "Laptops", "presupuesto_usd": 800})
    assert r.status_code == 401


def test_05_security_invalid_api_key_403() -> None:
    """TC-05 (Seguridad): Petición con X-API-Key errónea debe ser rechazada con HTTP 403."""
    r = client.post(
        "/predecir",
        headers={"X-API-Key": "clave-invalida-xyz"},
        json={"user_id": "u1", "categoria_preferida": "Laptops", "presupuesto_usd": 800},
    )
    assert r.status_code == 403


def test_06_security_rate_limiting_429() -> None:
    """TC-06 (Seguridad): Rate Limiter bloquea exceso de solicitudes con HTTP 429."""
    rate_limiter.set_limit(3)
    rate_limiter.reset()
    payload = {"user_id": "rl_user", "categoria_preferida": "Audio", "presupuesto_usd": 200}

    for _ in range(3):
        resp = client.post("/predecir", headers=HEADERS, json=payload)
        assert resp.status_code == 200

    blocked = client.post("/predecir", headers=HEADERS, json=payload)
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers


def test_07_input_validation_422() -> None:
    """TC-07 (Validación): Pydantic rechaza valores inválidos (presupuesto negativo / top_k > 10)."""
    r = client.post(
        "/predecir",
        headers=HEADERS,
        json={"user_id": "u2", "categoria_preferida": "Gaming", "presupuesto_usd": -50.0, "top_k": 99},
    )
    assert r.status_code == 422


def test_08_interpretability_and_ley_19628_privacy() -> None:
    """TC-08 (Negocio y Privacidad): Anonimización SHA-256 sin PII e interpretabilidad de recomendaciones."""
    raw_user = "araceli.salgado.rut123"
    r = client.post(
        "/predecir",
        headers=HEADERS,
        json={
            "user_id": raw_user,
            "categoria_preferida": "Smartphones",
            "presupuesto_usd": 750.0,
            "es_cyberday": True,
            "top_k": 4,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert raw_user not in data["user_hash_anonimizado"]
    assert data["user_hash_anonimizado"].startswith("usr_")
    assert "19.628" in data["cumplimiento_privacidad"]
    for rec in data["recomendaciones"]:
        assert len(rec["explicacion_negocio"]) >= 1
        assert rec["estrategia_comercial"] in ("Alta Conversión", "Upsell", "Cross-sell")
        assert "impacto_ticket_usd" in rec


def test_09_in_memory_cache_hit() -> None:
    """TC-09 (Rendimiento): Segunda consulta equivalente resuelve desde caché en memoria (< 5 ms)."""
    service.clear_cache()
    payload = {
        "user_id": "cache_tester",
        "categoria_preferida": "Monitores",
        "presupuesto_usd": 450.0,
        "vistas_previas_categoria": 4,
        "dias_desde_ultima_compra": 20,
        "items_en_carrito": 1,
        "es_cyberday": False,
        "top_k": 3,
    }
    first = client.post("/predecir", headers=HEADERS, json=payload).json()
    second = client.post("/predecir", headers=HEADERS, json=payload).json()
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["origen_inferencia"] == "in_memory_cache_hit"


def test_10_circuit_breaker_heuristic_fallback() -> None:
    """TC-10 (Resiliencia): Ante falla simulada se activa Circuit Breaker entregando Fallback sin HTTP 500."""
    r = client.post(
        "/predecir",
        headers=HEADERS,
        json={
            "user_id": "cb_user",
            "categoria_preferida": "Gaming",
            "presupuesto_usd": 300.0,
            "top_k": 3,
            "simular_fallo": True,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["circuit_breaker_activo"] is True
    assert data["origen_inferencia"] == "circuit_breaker_fallback"
    assert len(data["recomendaciones"]) == 3
    assert data["latencia_inferencia_ms"] < 25.0


def test_11_get_recommendations_by_user_endpoint() -> None:
    """TC-11 (Integración REST): Endpoint GET /recommendations/{user_id} responde con formato estándar."""
    r = client.get(
        "/recommendations/cliente_505?categoria=Audio&presupuesto_usd=350&top_k=3",
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["recomendaciones"]) == 3
    assert body["estado"] == "éxito"
    assert "X-Correlation-ID" in r.headers
    assert "Strict-Transport-Security" in r.headers


def test_12_e2e_interoperability_queue_worker_and_metrics() -> None:
    """TC-12 (Interoperabilidad E2E): API publica clickstream en cola -> Worker procesa -> /metrics actualiza."""
    ev_resp = client.post(
        "/events/clickstream",
        headers=HEADERS,
        json={
            "user_id": "e2e_buyer_77",
            "producto_id": "SF-LAP-0001",
            "accion": "add_to_cart",
            "precio_usd": 899.0,
        },
    )
    assert ev_resp.status_code == 200
    assert ev_resp.json()["estado"] == "encolado"

    consumer = EventConsumer()
    summary = consumer.process_batch_once()
    assert summary["total_events_processed"] >= 1
    assert Path(get_settings().processed_telemetry_path).exists()

    m_resp = client.get("/metrics")
    assert m_resp.status_code == 200
    metrics = m_resp.json()
    assert metrics["red_metrics"]["total_requests"] >= 1
    assert metrics["mensajeria"]["events_published"] >= 1
