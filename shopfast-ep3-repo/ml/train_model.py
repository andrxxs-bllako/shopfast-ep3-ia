"""
Script de entrenamiento y evaluación del modelo de recomendación de ShopFast (EP3).
Entrena un GradientBoostingClassifier sobre las interacciones históricas,
calcula métricas oficiales de clasificación (accuracy, precision, recall, F1, ROC-AUC),
mide latencia de inferencia, guarda el artefacto serializado y exporta gráficos de evidencia.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data.generate_dataset import FEATURE_COLUMNS, main as generate_data_main

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "shopfast_interactions.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "shopfast_recommender.joblib"
META_PATH = MODELS_DIR / "model_metadata.json"
EVIDENCE_CHARTS_DIR = BASE_DIR.parent / "evidencias" / "graficos"

HYPERPARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.08,
    "max_depth": 4,
    "subsample": 0.85,
    "min_samples_leaf": 20,
    "random_state": 42,
}

FEATURE_LABELS_ES = {
    "affinity_categoria": "Afinidad de categoría",
    "ratio_precio_presupuesto": "Relación precio / presupuesto",
    "rating_producto": "Calificación del producto",
    "descuento_pct": "Porcentaje de descuento",
    "vistas_previas_categoria": "Vistas previas en categoría",
    "dias_desde_ultima_compra": "Días desde última compra",
    "items_en_carrito": "Ítems en carrito",
    "es_cyberday": "Evento CyberDay",
    "popularidad_normalizada": "Popularidad normalizada",
    "compatibilidad_historial": "Compatibilidad con historial",
}


def save_confusion_matrix_chart(cm: np.ndarray, out_path: Path) -> None:
    """Genera imagen PNG de la matriz de confusión en alta resolución."""
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    classes = ["No Comprado (0)", "Comprado (1)"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Etiqueta Real",
        xlabel="Predicción del Modelo",
        title="Matriz de Confusión — Modelo ShopFast (Test 20%)",
    )
    ax.title.set_fontsize(11.5)
    ax.title.set_fontweight("bold")

    thresh = cm.max() / 2.0
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total) * 100.0
            ax.text(
                j,
                i,
                f"{val:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if val > thresh else "#1B365D",
                fontsize=10.5,
                fontweight="bold",
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_feature_importance_chart(importances: dict[str, float], out_path: Path) -> None:
    """Genera imagen PNG del ranking de importancia de variables para interpretabilidad."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=False)
    labels = [FEATURE_LABELS_ES.get(k, k) for k, _ in sorted_items]
    values = [v * 100.0 for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=300)
    colors = ["#1B365D" if v >= 8.0 else "#3B82F6" for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor="#0F172A", linewidth=0.5)
    ax.set_xlabel("Importancia Relativa (%)", fontsize=10)
    ax.set_title("Importancia de Variables (IE2)", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    ax.set_xlim(0, max(values) * 1.18)

    for bar, val in zip(bars, values):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.8, color="#0F172A", fontweight="bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def train_and_evaluate() -> dict:
    """Entrena el modelo, evalúa métricas en hold-out 20% y persiste artefactos."""
    if not DATA_PATH.exists():
        generate_data_main()

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["comprado"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(**HYPERPARAMS)
    t0 = time.perf_counter()
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    train_seconds = round(time.perf_counter() - t0, 3)

    X_test_np = X_test.to_numpy()
    y_pred = model.predict(X_test_np)
    y_prob = model.predict_proba(X_test_np)[:, 1]

    # Benchmark de latencia batch para rankear 30 productos candidatos (escenario real /predecir)
    sample_batch = X_test_np[:30]
    latencies_ms = []
    for _ in range(50):
        s0 = time.perf_counter()
        _ = model.predict_proba(sample_batch)
        latencies_ms.append((time.perf_counter() - s0) * 1000.0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    importances = {
        feat: round(float(imp), 5)
        for feat, imp in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    metadata = {
        "model_name": "ShopFast-GBDT-Recommender",
        "model_type": "GradientBoostingClassifier (Ranking y Probabilidad de Conversión)",
        "framework": "scikit-learn",
        "version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": "data/shopfast_interactions.csv",
            "total_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_rate": round(float(y.mean()), 4),
            "features": FEATURE_COLUMNS,
        },
        "hyperparameters": HYPERPARAMS,
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        },
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "training_time_seconds": train_seconds,
        "inference_latency_30_candidates_ms": {
            "p50": round(float(np.percentile(latencies_ms, 50)), 3),
            "p95": round(float(np.percentile(latencies_ms, 95)), 3),
            "mean": round(float(np.mean(latencies_ms)), 3),
        },
        "feature_importances": importances,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    save_confusion_matrix_chart(cm, EVIDENCE_CHARTS_DIR / "matriz_confusion.png")
    save_feature_importance_chart(importances, EVIDENCE_CHARTS_DIR / "importancia_caracteristicas.png")

    print("Entrenamiento finalizado con éxito:")
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Latencia inferencia (30 candidatos): p50={metadata['inference_latency_30_candidates_ms']['p50']} ms, p95={metadata['inference_latency_30_candidates_ms']['p95']} ms")
    print(f"Artefactos guardados en: {MODEL_PATH} y {META_PATH}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
