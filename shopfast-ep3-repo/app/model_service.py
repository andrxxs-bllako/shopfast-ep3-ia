"""Motor de inferencia, caché en memoria, SageMaker opcional, Circuit Breaker e interpretabilidad."""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.config import get_settings

FEATURE_COLUMNS = [
    "affinity_categoria",
    "ratio_precio_presupuesto",
    "rating_producto",
    "descuento_pct",
    "vistas_previas_categoria",
    "dias_desde_ultima_compra",
    "items_en_carrito",
    "es_cyberday",
    "popularidad_normalizada",
    "compatibilidad_historial",
]

COMPLEMENTARY_CATEGORIES = {
    "Laptops": ["Accesorios", "Monitores", "Almacenamiento"],
    "Smartphones": ["Audio", "Accesorios", "Redes"],
    "Audio": ["Smartphones", "Gaming", "Accesorios"],
    "Gaming": ["Monitores", "Audio", "Accesorios"],
    "Monitores": ["Laptops", "Gaming", "Accesorios"],
    "Almacenamiento": ["Laptops", "Accesorios", "Redes"],
    "Redes": ["Almacenamiento", "Laptops", "Smartphones"],
    "Accesorios": ["Laptops", "Smartphones", "Gaming"],
}


class RecommenderService:
    """Servicio de recomendación ML con caché TTL, invocación SageMaker y Fallback heurístico."""

    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.model_path = model_path or settings.model_path
        self.metadata_path = metadata_path or settings.metadata_path
        self.catalog_path = catalog_path or settings.catalog_path

        self.model = joblib.load(self.model_path)
        self.metadata: dict[str, Any] = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.catalog: list[dict[str, Any]] = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.popular_catalog = sorted(self.catalog, key=lambda p: (p["popularidad_normalizada"], p["rating_producto"]), reverse=True)

        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.fallback_count = 0
        self.sagemaker_calls = 0
        self.circuit_state = "CLOSED"

    def clear_cache(self) -> None:
        """Limpia la caché en memoria."""
        with self._lock:
            self._cache.clear()

    def _build_cache_key(self, params: dict[str, Any]) -> str:
        key_parts = [
            str(params["categoria_preferida"]),
            str(round(float(params["presupuesto_usd"]), 0)),
            str(int(params["vistas_previas_categoria"])),
            str(int(params["items_en_carrito"])),
            str(int(params["es_cyberday"])),
            str(int(params["top_k"])),
        ]
        return "|".join(key_parts)

    def _select_candidates(self, categoria_preferida: str, limit: int = 36) -> list[dict[str, Any]]:
        primary = [p for p in self.catalog if p["categoria"].lower() == categoria_preferida.lower()]
        complements = COMPLEMENTARY_CATEGORIES.get(categoria_preferida, ["Accesorios"])
        secondary = [p for p in self.catalog if p["categoria"] in complements]
        if not primary:
            return self.popular_catalog[:limit]
        return (primary[:24] + secondary[:12])[:limit]

    def _build_feature_matrix(self, candidates: list[dict[str, Any]], req: dict[str, Any]) -> np.ndarray:
        budget = max(float(req["presupuesto_usd"]), 20.0)
        cat_pref = req["categoria_preferida"].lower()
        vistas = int(req["vistas_previas_categoria"])
        dias = int(req["dias_desde_ultima_compra"])
        carrito = int(req["items_en_carrito"])
        cyber = 1 if req["es_cyberday"] else 0

        rows = []
        for item in candidates:
            is_primary = item["categoria"].lower() == cat_pref
            affinity = 0.88 if is_primary else 0.56
            compat = min(0.98, 0.50 * affinity + 0.30 * item["popularidad_normalizada"] + 0.20 * (1.0 if carrito > 0 else 0.5))
            ratio_precio = min(2.40, max(0.15, float(item["precio_usd"]) / budget))
            rows.append(
                [
                    affinity,
                    ratio_precio,
                    float(item["rating_producto"]),
                    float(item["descuento_pct"]),
                    float(vistas if is_primary else max(1, vistas // 2)),
                    float(dias),
                    float(carrito),
                    float(cyber),
                    float(item["popularidad_normalizada"]),
                    compat,
                ]
            )
        return np.array(rows, dtype=float)

    def _invoke_sagemaker_or_local(self, features: np.ndarray) -> tuple[np.ndarray, str]:
        """Invoca endpoint AWS SageMaker si está configurado; de lo contrario usa modelo local."""
        settings = get_settings()
        if settings.sagemaker_endpoint_name:
            try:
                import boto3

                runtime = boto3.client("sagemaker-runtime", region_name=settings.aws_region)
                payload = json.dumps({"instances": features.tolist()})
                resp = runtime.invoke_endpoint(
                    EndpointName=settings.sagemaker_endpoint_name,
                    ContentType="application/json",
                    Body=payload,
                )
                body = json.loads(resp["Body"].read().decode("utf-8"))
                probs = np.array(body["predictions"], dtype=float)
                self.sagemaker_calls += 1
                return probs, f"sagemaker:{settings.sagemaker_endpoint_name}"
            except Exception:
                pass

        probs = self.model.predict_proba(features)[:, 1]
        return probs, "local-gbdt-sklearn"

    def _explain_product(self, item: dict[str, Any], prob: float, req: dict[str, Any]) -> dict[str, Any]:
        """Genera explicación de negocio interpretable para toma de decisiones (IE2)."""
        reasons: list[str] = []
        is_primary = item["categoria"].lower() == req["categoria_preferida"].lower()
        budget = float(req["presupuesto_usd"])
        precio = float(item["precio_usd"])

        if is_primary:
            reasons.append(f"Coincide directamente con categoría de interés ({item['categoria']})")
        else:
            reasons.append(f"Complemento estratégico Cross-sell en categoría {item['categoria']}")

        if precio <= budget:
            reasons.append(f"Dentro del presupuesto objetivo (USD {precio:.0f} <= USD {budget:.0f})")
        elif precio <= budget * 1.2:
            reasons.append(f"Oportunidad Upsell cercana al presupuesto (USD {precio:.0f})")

        if item["rating_producto"] >= 4.4:
            reasons.append(f"Alta satisfacción de compradores ({item['rating_producto']}★)")

        if req["es_cyberday"] and item["descuento_pct"] >= 15:
            reasons.append(f"Promoción CyberDay activa ({item['descuento_pct']}% dcto)")
        elif item["descuento_pct"] >= 15:
            reasons.append(f"Descuento comercial atractivo ({item['descuento_pct']}%)")

        if not is_primary:
            estrategia = "Cross-sell"
        elif precio > budget * 0.95:
            estrategia = "Upsell"
        else:
            estrategia = "Alta Conversión"

        return {
            "producto_id": item["producto_id"],
            "nombre": item["nombre"],
            "categoria": item["categoria"],
            "precio_usd": item["precio_usd"],
            "descuento_pct": item["descuento_pct"],
            "rating_producto": item["rating_producto"],
            "score_afinidad": round(float(prob), 4),
            "nivel_confianza": "Alto" if prob >= 0.72 else "Medio",
            "estrategia_comercial": estrategia,
            "impacto_ticket_usd": round(precio - 85.0, 2),
            "explicacion_negocio": reasons[:3],
        }

    def _heuristic_fallback(self, top_k: int, categoria_preferida: str) -> list[dict[str, Any]]:
        """Catálogo heurístico de respaldo en < 8 ms ante caída o latencia > 180 ms (Circuit Breaker)."""
        cat_items = [p for p in self.popular_catalog if p["categoria"].lower() == categoria_preferida.lower()]
        pool = cat_items if len(cat_items) >= top_k else self.popular_catalog
        selected = pool[:top_k]
        results = []
        for item in selected:
            results.append(
                {
                    "producto_id": item["producto_id"],
                    "nombre": item["nombre"],
                    "categoria": item["categoria"],
                    "precio_usd": item["precio_usd"],
                    "descuento_pct": item["descuento_pct"],
                    "rating_producto": item["rating_producto"],
                    "score_afinidad": round(float(item["popularidad_normalizada"]), 4),
                    "nivel_confianza": "Respaldo Heurístico",
                    "estrategia_comercial": "Top Ventas Garantizado",
                    "impacto_ticket_usd": round(float(item["precio_usd"]) - 85.0, 2),
                    "explicacion_negocio": [
                        "Activación de Circuit Breaker: entrega degradada elegante sin error HTTP 500",
                        f"Producto más popular en {item['categoria']} ({item['rating_producto']}★)",
                    ],
                }
            )
        return results

    def recommend(self, req: dict[str, Any]) -> dict[str, Any]:
        """Ejecuta recomendación con soporte de caché, inferencia ML y Circuit Breaker."""
        settings = get_settings()
        t0 = time.perf_counter()
        top_k = int(req.get("top_k", 5))
        cat_pref = req.get("categoria_preferida", "Laptops")

        if req.get("simular_fallo"):
            self.fallback_count += 1
            self.circuit_state = "OPEN (Fallback Activo)"
            recs = self._heuristic_fallback(top_k, cat_pref)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return {
                "recomendaciones": recs,
                "origen_inferencia": "circuit_breaker_fallback",
                "cache_hit": False,
                "circuit_breaker_activo": True,
                "latencia_inferencia_ms": elapsed_ms,
            }

        cache_key = self._build_cache_key(req)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (now - cached[0]) < settings.cache_ttl_seconds:
                self.cache_hits += 1
                self.circuit_state = "CLOSED"
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                return {
                    "recomendaciones": cached[1][:top_k],
                    "origen_inferencia": "in_memory_cache_hit",
                    "cache_hit": True,
                    "circuit_breaker_activo": False,
                    "latencia_inferencia_ms": elapsed_ms,
                }

        try:
            candidates = self._select_candidates(cat_pref)
            matrix = self._build_feature_matrix(candidates, req)
            probs, engine_used = self._invoke_sagemaker_or_local(matrix)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if elapsed_ms > settings.circuit_breaker_timeout_ms:
                raise TimeoutError(f"Inferencia excedió umbral {settings.circuit_breaker_timeout_ms} ms")

            ranked_idx = np.argsort(probs)[::-1][:top_k]
            recs = [self._explain_product(candidates[i], float(probs[i]), req) for i in ranked_idx]

            with self._lock:
                self.cache_misses += 1
                self._cache[cache_key] = (time.monotonic(), recs)
            self.circuit_state = "CLOSED"

            return {
                "recomendaciones": recs,
                "origen_inferencia": engine_used,
                "cache_hit": False,
                "circuit_breaker_activo": False,
                "latencia_inferencia_ms": round(elapsed_ms, 3),
            }
        except Exception:
            self.fallback_count += 1
            self.circuit_state = "OPEN (Fallback Activo)"
            recs = self._heuristic_fallback(top_k, cat_pref)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return {
                "recomendaciones": recs,
                "origen_inferencia": "circuit_breaker_fallback",
                "cache_hit": False,
                "circuit_breaker_activo": True,
                "latencia_inferencia_ms": elapsed_ms,
            }
