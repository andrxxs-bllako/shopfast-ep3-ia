"""Capa de comunicación distribuida y mensajería asíncrona (Amazon SQS / RabbitMQ / Local)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings


class EventPublisher:
    """Publica eventos de recomendación y clickstream de forma desacoplada."""

    def __init__(self, local_path: Path | None = None) -> None:
        settings = get_settings()
        self.backend = settings.queue_backend
        self.local_path = local_path or settings.local_queue_path
        self.sqs_queue_url = settings.sqs_queue_url
        self.aws_region = settings.aws_region
        self.rabbitmq_url = settings.rabbitmq_url
        self.rabbitmq_queue = settings.rabbitmq_queue
        self._lock = threading.Lock()
        self.published_count = 0
        self.last_backend_used = self.backend

    def publish(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Envía evento al broker configurado con respaldo local ante indisponibilidad."""
        envelope = {
            "event_type": event_type,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        delivered_via = "local"
        if self.backend == "sqs" and self.sqs_queue_url:
            if self._publish_sqs(envelope):
                delivered_via = "sqs"
        elif self.backend == "rabbitmq":
            if self._publish_rabbitmq(envelope):
                delivered_via = "rabbitmq"

        if delivered_via == "local":
            self._publish_local(envelope)

        with self._lock:
            self.published_count += 1
            self.last_backend_used = delivered_via

        return {"delivered": True, "backend": delivered_via, "event_type": event_type}

    def _publish_sqs(self, envelope: dict[str, Any]) -> bool:
        try:
            import boto3

            client = boto3.client("sqs", region_name=self.aws_region)
            client.send_message(QueueUrl=self.sqs_queue_url, MessageBody=json.dumps(envelope, ensure_ascii=False))
            return True
        except Exception:
            return False

    def _publish_rabbitmq(self, envelope: dict[str, Any]) -> bool:
        try:
            import pika

            params = pika.URLParameters(self.rabbitmq_url)
            params.socket_timeout = 1.5
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=self.rabbitmq_queue, durable=True)
            channel.basic_publish(
                exchange="",
                routing_key=self.rabbitmq_queue,
                body=json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2),
            )
            connection.close()
            return True
        except Exception:
            return False

    def _publish_local(self, envelope: dict[str, Any]) -> None:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(envelope, ensure_ascii=False)
        with self._lock:
            with self.local_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")

    def read_local_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Lee los últimos eventos persistidos en la cola local."""
        if not self.local_path.exists():
            return []
        lines = self.local_path.read_text(encoding="utf-8").splitlines()
        events = []
        for raw in lines[-limit:]:
            if raw.strip():
                events.append(json.loads(raw))
        return events


class EventConsumer:
    """Consumidor de eventos asíncronos desde SQS, RabbitMQ o cola local."""

    def __init__(self, local_path: Path | None = None, output_path: Path | None = None) -> None:
        settings = get_settings()
        self.backend = settings.queue_backend
        self.local_path = local_path or settings.local_queue_path
        self.output_path = output_path or settings.processed_telemetry_path
        self.sqs_queue_url = settings.sqs_queue_url
        self.aws_region = settings.aws_region
        self.rabbitmq_url = settings.rabbitmq_url
        self.rabbitmq_queue = settings.rabbitmq_queue

    def process_batch_once(self) -> dict[str, Any]:
        """Consume un lote disponible de eventos y actualiza el resumen de telemetría."""
        events: list[dict[str, Any]] = []
        source_used = self.backend

        if self.backend == "sqs" and self.sqs_queue_url:
            events = self._consume_sqs()
        elif self.backend == "rabbitmq":
            events = self._consume_rabbitmq()
        else:
            events = self._consume_local()
            source_used = "local"

        summary = self._update_telemetry(events, source_used)
        return summary

    def _consume_local(self) -> list[dict[str, Any]]:
        if not self.local_path.exists():
            return []
        lines = [ln for ln in self.local_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines]

    def _consume_sqs(self) -> list[dict[str, Any]]:
        try:
            import boto3

            client = boto3.client("sqs", region_name=self.aws_region)
            resp = client.receive_message(QueueUrl=self.sqs_queue_url, MaxNumberOfMessages=10, WaitTimeSeconds=1)
            events = []
            for msg in resp.get("Messages", []):
                events.append(json.loads(msg["Body"]))
                client.delete_message(QueueUrl=self.sqs_queue_url, ReceiptHandle=msg["ReceiptHandle"])
            return events
        except Exception:
            return self._consume_local()

    def _consume_rabbitmq(self) -> list[dict[str, Any]]:
        try:
            import pika

            params = pika.URLParameters(self.rabbitmq_url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=self.rabbitmq_queue, durable=True)
            events = []
            for _ in range(50):
                method, _props, body = channel.basic_get(queue=self.rabbitmq_queue, auto_ack=True)
                if not method:
                    break
                events.append(json.loads(body.decode("utf-8")))
            connection.close()
            return events
        except Exception:
            return self._consume_local()

    def _update_telemetry(self, events: list[dict[str, Any]], source_used: str) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        categories: dict[str, int] = {}
        for ev in events:
            et = ev.get("event_type", "unknown")
            by_type[et] = by_type.get(et, 0) + 1
            cat = ev.get("payload", {}).get("categoria_preferida")
            if cat:
                categories[cat] = categories.get(cat, 0) + 1

        summary = {
            "last_processed_utc": datetime.now(timezone.utc).isoformat(),
            "source_backend": source_used,
            "total_events_processed": len(events),
            "events_by_type": by_type,
            "top_categories": categories,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

