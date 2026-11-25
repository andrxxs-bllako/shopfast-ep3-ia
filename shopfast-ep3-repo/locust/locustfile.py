"""Prueba de carga y estrés con Locust para ShopFast API REST (EP3 — IE10)."""

from __future__ import annotations

import os
import random
from locust import HttpUser, between, task

API_KEY = os.getenv("SHOPFAST_API_KEY", "shopfast-ep3-key-2026")
CATEGORIES = ["Laptops", "Smartphones", "Audio", "Gaming", "Monitores", "Almacenamiento", "Redes", "Accesorios"]


class ShopFastUser(HttpUser):
    """Simula compradores concurrentes solicitando recomendaciones en tiempo real."""

    wait_time = between(0.3, 1.0)

    def on_start(self) -> None:
        self.headers = {"X-API-Key": API_KEY}

    @task(4)
    def post_predecir(self) -> None:
        cat = random.choice(CATEGORIES)
        payload = {
            "user_id": f"user_{random.randint(1, 5000)}",
            "categoria_preferida": cat,
            "presupuesto_usd": random.choice([250.0, 550.0, 950.0, 1400.0]),
            "vistas_previas_categoria": random.randint(1, 10),
            "dias_desde_ultima_compra": random.randint(1, 90),
            "items_en_carrito": random.randint(0, 3),
            "es_cyberday": random.choice([True, False]),
            "top_k": 5,
        }
        self.client.post("/predecir", json=payload, headers=self.headers, name="/predecir")

    @task(1)
    def get_recommendations(self) -> None:
        uid = f"user_{random.randint(1, 5000)}"
        cat = random.choice(CATEGORIES)
        self.client.get(
            f"/recommendations/{uid}?categoria={cat}&presupuesto_usd=850&top_k=5",
            headers=self.headers,
            name="/recommendations/{user_id}",
        )
