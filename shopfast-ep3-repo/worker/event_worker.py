"""Worker asíncrono para consumir eventos de SQS / RabbitMQ / cola local (IE7)."""

from __future__ import annotations

import argparse
import json
import time

from app.messaging import EventConsumer


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker de ingesta asíncrona ShopFast")
    parser.add_argument("--once", action="store_true", help="Procesa un lote y finaliza.")
    parser.add_argument("--interval", type=float, default=5.0, help="Segundos entre ciclos en modo continuo.")
    args = parser.parse_args()

    consumer = EventConsumer()
    if args.once:
        summary = consumer.process_batch_once()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    print(f"Worker iniciado (backend={consumer.backend}). Esperando eventos...")
    while True:
        summary = consumer.process_batch_once()
        print(f"[Worker] Procesados={summary['total_events_processed']} backend={summary['source_backend']}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
