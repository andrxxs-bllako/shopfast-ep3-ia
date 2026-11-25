"""Controles de seguridad API REST (IE7): API Key, Rate Limiting, TLS headers y Ley N° 19.628."""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from app.config import get_settings


class SlidingWindowRateLimiter:
    """Limitador de tasa por ventana móvil de 60 segundos en memoria (thread-safe)."""

    def __init__(self, max_requests_per_minute: int = 120) -> None:
        self.max_requests = max_requests_per_minute
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def set_limit(self, limit: int) -> None:
        """Actualiza dinámicamente el límite por minuto (útil en pruebas y configuración)."""
        with self._lock:
            self.max_requests = max(1, int(limit))

    def reset(self) -> None:
        """Limpia contadores de ventana para pruebas unitarias aisladas."""
        with self._lock:
            self._buckets.clear()

    def check(self, client_id: str) -> tuple[bool, int, int]:
        """Verifica si el cliente tiene cuota disponible en la ventana de 60s."""
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            q = self._buckets[client_id]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                return False, 0, self.max_requests
            q.append(now)
            remaining = max(0, self.max_requests - len(q))
            return True, remaining, self.max_requests


rate_limiter = SlidingWindowRateLimiter(max_requests_per_minute=get_settings().rate_limit_rpm)


def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """Valida cabecera X-API-Key y aplica control de Rate Limiting por cliente."""
    settings = get_settings()
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Falta cabecera de autenticación X-API-Key.",
        )

    if not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Credencial X-API-Key inválida o revocada.",
        )

    client_host = request.client.host if request.client else "unknown"
    client_key = f"{client_host}:{x_api_key[:8]}"
    allowed, remaining, limit = rate_limiter.check(client_key)
    request.state.rate_limit = limit
    request.state.rate_remaining = remaining

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Límite de peticiones excedido (Rate Limiting activo). Reintente en unos segundos.",
            headers={"Retry-After": "60", "X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"},
        )

    return x_api_key


def anonymize_user_id(user_id: str, salt: str | None = None) -> str:
    """Anonimiza el identificador del usuario con SHA-256 + salt (Ley Chilena N° 19.628)."""
    effective_salt = salt or get_settings().anonymization_salt
    digest = hashlib.sha256(f"{user_id}:{effective_salt}".encode("utf-8")).hexdigest()
    return f"usr_{digest[:16]}"
