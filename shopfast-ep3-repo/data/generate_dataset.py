"""
Generador de catálogo de productos y dataset histórico de interacciones para ShopFast (EP3).
Produce datos sintéticos realistas y reproducibles (seed=42) alineados al Caso 1:
- 240 productos tecnológicos reales distribuidos en 8 categorías (precios USD 19 - 2.499).
- 20.000 interacciones usuario-producto con 10 variables predictoras de afinidad comercial.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_INTERACTIONS = 20_000
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

CATEGORIES_SPEC: dict[str, dict] = {
    "Laptops": {
        "brands": ["Lenovo", "ASUS", "HP", "Apple", "Dell", "Acer"],
        "models": ["ThinkPad X1", "ROG Zephyrus", "Pavilion Plus", "MacBook Air M3", "XPS 13", "Swift Go"],
        "specs": ["16GB/512GB", "32GB/1TB", "16GB/1TB", "8GB/256GB", "32GB/2TB"],
        "price_range": (649.0, 2499.0),
        "margin_range": (0.14, 0.26),
    },
    "Smartphones": {
        "brands": ["Samsung", "Apple", "Xiaomi", "Google", "Motorola", "OnePlus"],
        "models": ["Galaxy S25", "iPhone 16", "Redmi Note Pro", "Pixel 9", "Edge 50", "13R 5G"],
        "specs": ["128GB", "256GB", "512GB", "256GB 5G", "128GB Dual"],
        "price_range": (249.0, 1499.0),
        "margin_range": (0.15, 0.28),
    },
    "Audio": {
        "brands": ["Sony", "Bose", "JBL", "Sennheiser", "HyperX", "Apple"],
        "models": ["WH-1000XM5", "QuietComfort Ultra", "Tune 770NC", "Momentum 4", "Cloud III", "AirPods Pro 2"],
        "specs": ["ANC Bluetooth", "Hi-Res USB-C", "Wireless Low-Latency", "Studio Edition", "Sport IPX5"],
        "price_range": (49.0, 449.0),
        "margin_range": (0.28, 0.45),
    },
    "Gaming": {
        "brands": ["Logitech", "Razer", "PlayStation", "Nintendo", "SteelSeries", "Corsair"],
        "models": ["G Pro X Superlight", "BlackWidow V4", "DualSense Edge", "Switch OLED Dock", "Apex Pro", "K70 RGB"],
        "specs": ["Wireless 8K", "Mechanical Red", "Pro Controller", "Edición Torneo", "Hall Effect"],
        "price_range": (59.0, 799.0),
        "margin_range": (0.22, 0.40),
    },
    "Monitores": {
        "brands": ["LG", "Samsung", "ASUS", "Dell", "AOC", "BenQ"],
        "models": ["UltraGear 27\"", "Odyssey G5 32\"", "ProArt 27\"", "UltraSharp 4K", "Hero 24\"", "Mobius 27\""],
        "specs": ["144Hz IPS", "165Hz Curvo", "4K USB-C 90W", "180Hz QHD", "240Hz OLED"],
        "price_range": (169.0, 1199.0),
        "margin_range": (0.18, 0.32),
    },
    "Almacenamiento": {
        "brands": ["Samsung", "WD", "Kingston", "Crucial", "Seagate", "SanDisk"],
        "models": ["990 PRO NVMe", "Black SN850X", "KC3000 PCIe4", "T7 Shield SSD", "IronWolf NAS", "Extreme Portable"],
        "specs": ["1TB", "2TB", "4TB", "500GB", "2TB USB 3.2"],
        "price_range": (45.0, 499.0),
        "margin_range": (0.24, 0.38),
    },
    "Redes": {
        "brands": ["TP-Link", "ASUS", "Ubiquiti", "Netgear", "MikroTik", "D-Link"],
        "models": ["Archer AX73", "RT-AX88U Pro", "UniFi U6 Pro", "Nighthawk WiFi 7", "Deco XE75 Mesh", "hAP ax3"],
        "specs": ["WiFi 6 AX5400", "WiFi 6E Tri-Band", "PoE+ Dual Band", "Pack 2 Nodos", "2.5G Multi-Gig"],
        "price_range": (39.0, 549.0),
        "margin_range": (0.25, 0.42),
    },
    "Accesorios": {
        "brands": ["Anker", "Logitech", "UGREEN", "Belkin", "Keychron", "Elgato"],
        "models": ["GaN Prime 100W", "MX Master 3S", "Hub USB-C 9-en-1", "Base MagSafe 3en1", "K2 Pro Wireless", "Stream Deck MK.2"],
        "specs": ["Carga Rápida", "Ergonómico", "4K60 HDMI", "Aluminio CNC", "USB-C Braided"],
        "price_range": (19.0, 229.0),
        "margin_range": (0.35, 0.55),
    },
}

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


def generate_catalog(seed: int = SEED) -> list[dict]:
    """Genera el catálogo de 240 productos tecnológicos para ShopFast."""
    rng = np.random.default_rng(seed)
    catalog: list[dict] = []
    prod_idx = 1

    for category, spec in CATEGORIES_SPEC.items():
        low_p, high_p = spec["price_range"]
        low_m, high_m = spec["margin_range"]
        for i in range(30):
            pair_idx = i % len(spec["brands"])
            brand = spec["brands"][pair_idx]
            model = spec["models"][pair_idx % len(spec["models"])]
            variant = spec["specs"][(i // len(spec["brands"])) % len(spec["specs"])]
            precio = round(float(rng.uniform(low_p, high_p)), 2)
            rating = round(float(np.clip(rng.normal(4.45, 0.32), 3.4, 5.0)), 2)
            descuento = int(rng.choice([0, 0, 5, 10, 15, 20, 25, 30, 35], p=[0.25, 0.15, 0.12, 0.15, 0.12, 0.08, 0.06, 0.04, 0.03]))
            popularidad = round(float(np.clip(rng.beta(2.4, 2.2), 0.12, 0.99)), 3)
            margen = round(float(rng.uniform(low_m, high_m)), 3)
            stock = int(rng.integers(8, 260))

            catalog.append(
                {
                    "producto_id": f"SF-{category[:3].upper()}-{prod_idx:04d}",
                    "nombre": f"{brand} {model} ({variant})",
                    "categoria": category,
                    "marca": brand,
                    "precio_usd": precio,
                    "rating_producto": rating,
                    "descuento_pct": descuento,
                    "popularidad_normalizada": popularidad,
                    "margen_estimado": margen,
                    "stock": stock,
                }
            )
            prod_idx += 1

    return catalog


def generate_interactions(n_rows: int = N_INTERACTIONS, seed: int = SEED) -> pd.DataFrame:
    """Genera dataset tabular de interacciones históricas con señal realista y ruido controlado."""
    rng = np.random.default_rng(seed + 1)

    affinity_categoria = np.round(np.clip(rng.beta(2.2, 2.0, n_rows), 0.02, 0.99), 4)
    ratio_precio_presupuesto = np.round(np.clip(rng.lognormal(mean=-0.05, sigma=0.42, size=n_rows), 0.15, 2.40), 4)
    rating_producto = np.round(np.clip(rng.normal(4.35, 0.38, n_rows), 3.20, 5.00), 2)
    descuento_pct = np.round(rng.choice([0, 5, 10, 15, 20, 25, 30, 40], size=n_rows, p=[0.30, 0.12, 0.16, 0.14, 0.11, 0.08, 0.06, 0.03]), 1)
    vistas_previas_categoria = rng.poisson(lam=3.8, size=n_rows).clip(0, 25)
    dias_desde_ultima_compra = rng.integers(0, 181, size=n_rows)
    items_en_carrito = rng.choice([0, 1, 2, 3, 4, 5], size=n_rows, p=[0.42, 0.25, 0.16, 0.09, 0.05, 0.03])
    es_cyberday = rng.choice([0, 1], size=n_rows, p=[0.78, 0.22])
    popularidad_normalizada = np.round(np.clip(rng.beta(2.3, 2.1, n_rows), 0.05, 0.99), 4)
    compatibilidad_historial = np.round(np.clip(0.55 * affinity_categoria + 0.45 * rng.beta(2.0, 2.0, n_rows), 0.02, 0.99), 4)

    # Ecuación latente de probabilidad de compra con interacciones no lineales (para árbol de decisión / GBDT)
    price_penalty = np.where(
        ratio_precio_presupuesto <= 1.05,
        1.35 * (1.05 - np.abs(ratio_precio_presupuesto - 0.78)),
        -2.60 * (ratio_precio_presupuesto - 1.05),
    )
    recency_boost = -0.011 * dias_desde_ultima_compra + np.where(dias_desde_ultima_compra <= 21, 0.55, 0.0)
    cyber_promo = es_cyberday * (0.045 * descuento_pct + 0.35)
    cart_signal = np.where(items_en_carrito >= 1, 0.42 + 0.14 * np.minimum(items_en_carrito, 4), -0.30)

    logit = (
        -3.55
        + 3.60 * affinity_categoria
        + 1.35 * price_penalty
        + 0.85 * (rating_producto - 4.0)
        + 0.042 * descuento_pct
        + 0.18 * np.minimum(vistas_previas_categoria, 10)
        + 1.25 * recency_boost
        + 1.25 * cart_signal
        + 1.25 * cyber_promo
        + 1.65 * popularidad_normalizada
        + 1.95 * compatibilidad_historial
        + 1.05 * (affinity_categoria * compatibilidad_historial)
        + rng.normal(0.0, 0.28, size=n_rows)
    )

    prob = 1.0 / (1.0 + np.exp(-logit))
    comprado = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "affinity_categoria": affinity_categoria,
            "ratio_precio_presupuesto": ratio_precio_presupuesto,
            "rating_producto": rating_producto,
            "descuento_pct": descuento_pct,
            "vistas_previas_categoria": vistas_previas_categoria,
            "dias_desde_ultima_compra": dias_desde_ultima_compra,
            "items_en_carrito": items_en_carrito,
            "es_cyberday": es_cyberday,
            "popularidad_normalizada": popularidad_normalizada,
            "compatibilidad_historial": compatibilidad_historial,
            "comprado": comprado,
        }
    )
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    catalog = generate_catalog(SEED)
    catalog_path = MODELS_DIR / "product_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    df = generate_interactions(N_INTERACTIONS, SEED)
    dataset_path = DATA_DIR / "shopfast_interactions.csv"
    df.to_csv(dataset_path, index=False)

    pos_rate = float(df["comprado"].mean())
    print(f"Catálogo generado: {len(catalog)} productos -> {catalog_path}")
    print(f"Dataset generado: {len(df)} filas, tasa positiva={pos_rate:.2%} -> {dataset_path}")


if __name__ == "__main__":
    main()
