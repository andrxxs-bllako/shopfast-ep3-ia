# ShopFast AI — Sistema Integral de Recomendación Empresarial (EP3 — ITY1102)

Repositorio oficial del **Equipo N° 5** (*Araceli Salgado y Andrés Concha*) para la **Evaluación Parcial 3** de *Arquitectura de Sistemas de Inteligencia Artificial* (Duoc UC, Docente: Ricardo Alberto Aravena Videla).

## Arquitectura Implementada
- **Modelo ML Real (`ml/train_model.py`, `data/generate_dataset.py`):** `GradientBoostingClassifier` (`scikit-learn`) entrenado sobre 20.000 interacciones de e-commerce y catálogo de 240 productos tecnológicos en 8 categorías, con evaluación hold-out 80/20 (`accuracy`, `precision`, `recall`, `F1-score`, `ROC-AUC`).
- **API REST Empresarial (`app/main.py`):** FastAPI + OpenAPI 3.1 (`/docs`), autenticación por `X-API-Key`, Rate Limiting (`429`), anonimización de usuarios mediante `SHA-256 + salt` (Ley Chilena N° 19.628), caché TTL en memoria, **Circuit Breaker** con Fallback heurístico (`< 8 ms`), y trazabilidad (`X-Correlation-ID`, métricas RED en `/metrics`).
- **Interpretabilidad para Toma de Decisiones (`app/model_service.py`):** cada recomendación entrega `score_afinidad`, `nivel_confianza`, `estrategia_comercial` (`Alta Conversión`, `Upsell`, `Cross-sell`), `impacto_ticket_usd` y `explicacion_negocio`.
- **Comunicación Distribuida (`app/messaging.py`, `worker/event_worker.py`):** publicación asíncrona desacoplada compatible con **Amazon SQS**, **RabbitMQ** y cola persistente local.
- **Plataforma AIaaS AWS SageMaker (`sagemaker/`):** script de inferencia (`inference.py`) y despliegue automatizado (`deploy_sagemaker.py`) con Application Auto Scaling (1 a 4 instancias).
- **Dashboard Interactivo (`dashboard/app.py`):** interfaz Streamlit conectada en vivo a la API para visualizar recomendaciones interpretables, métricas del modelo y observabilidad RED.
- **Contenedorización y Seguridad Perimetral:** `Dockerfile` Multi-Stage Build con usuario no privilegiado y `HEALTHCHECK`, `docker-compose.yml` (API + Dashboard + Worker + RabbitMQ + perfil Nginx TLS 1.3).

## Estructura del Repositorio
```text
shopfast-ep3-repo/
├── app/
│   ├── config.py               # Configuración por variables de entorno
│   ├── main.py                 # API REST FastAPI y métricas RED
│   ├── messaging.py            # Publicador/Consumidor SQS, RabbitMQ y local
│   ├── model_service.py        # Inferencia ML/SageMaker, caché, Circuit Breaker e interpretabilidad
│   └── security.py             # X-API-Key, Rate Limiter, anonimización Ley 19.628
├── dashboard/app.py            # Dashboard interactivo Streamlit
├── data/generate_dataset.py    # Generación de catálogo (240) e interacciones (20.000)
├── ml/train_model.py           # Entrenamiento, métricas y gráficos de evaluación
├── sagemaker/
│   ├── inference.py            # Handler model_fn/input_fn/predict_fn/output_fn
│   └── deploy_sagemaker.py     # Empaquetado model.tar.gz, S3, Endpoint y Auto Scaling
├── worker/event_worker.py      # Worker consumidor asíncrono
├── locust/locustfile.py        # Pruebas de carga y estrés concurrente
├── nginx/nginx.conf            # Proxy inverso TLS 1.3 y rate limiting perimetral
├── postman/ShopFast_EP3.postman_collection.json
├── tests/test_api_and_model.py # 12 pruebas automatizadas (unitarias, seguridad, E2E)
├── Dockerfile
├── Dockerfile.dashboard
└── docker-compose.yml
```

## Ejecución Rápida Local
```bash
# 1. Instalar dependencias
pip install -r requirements-dev.txt

# 2. Generar dataset y entrenar modelo real
python -m data.generate_dataset
python -m ml.train_model

# 3. Ejecutar suite de 12 pruebas automatizadas
pytest -v tests/test_api_and_model.py

# 4. Iniciar API REST (puerto 8000) y Dashboard Streamlit (puerto 8501)
uvicorn app.main:app --host 0.0.0.0 --port 8000
streamlit run dashboard/app.py --server.port 8501
```

## Despliegue con Docker Compose (API + RabbitMQ + Worker + Dashboard)
```bash
docker compose up --build -d
# API Swagger: http://localhost:8000/docs
# Dashboard:   http://localhost:8501
# RabbitMQ UI: http://localhost:15672 (guest/guest)
```

## Prueba de Carga con Locust
```bash
locust -f locust/locustfile.py --host http://127.0.0.1:8000
```
