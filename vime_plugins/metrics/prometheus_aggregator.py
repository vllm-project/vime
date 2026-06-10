"""Unified Prometheus metrics endpoint, ported from ai21-verl.

Re-homes ``verl/ai21/utils/metrics_aggregator.py``. Serves a single ``/metrics`` endpoint
that aggregates two sources so production Prometheus only needs one scrape target per node:

1. **Multiprocess metrics** — vLLM engines (and any other process using ``prometheus_client``)
   write metric files into the shared ``PROMETHEUS_MULTIPROC_DIR`` directory; they are read
   back via ``MultiProcessCollector``.
2. **Ray metrics** — scraped from Ray's local metrics endpoint
   (``ray start --metrics-export-port <port>``).

Run it as a sidecar next to the training job (same pattern as
``vime_plugins/checkpoint/gcs_sync.py``)::

    export PROMETHEUS_MULTIPROC_DIR=/tmp/prom_multiproc && mkdir -p $PROMETHEUS_MULTIPROC_DIR
    ray start --head --metrics-export-port 8080 ...
    python -m vime_plugins.metrics.prometheus_aggregator --port 9090 --ray-metrics-port 8080 &

or start it programmatically with :func:`start_unified_metrics_server`.

Requires ``prometheus_client`` (vLLM already depends on it). ``PROMETHEUS_MULTIPROC_DIR``
must be set *before* Ray / vLLM start or their metric files land nowhere.
"""

import argparse
import logging
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)

__all__ = ["UnifiedMetricsHandler", "start_unified_metrics_server", "main"]


def _generate_multiprocess_metrics() -> bytes:
    # Imported lazily so the module is importable without prometheus_client.
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry)


class UnifiedMetricsHandler(BaseHTTPRequestHandler):
    """Serves /metrics by combining multiprocess metric files and Ray's endpoint."""

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        try:
            self.wfile.write(_generate_multiprocess_metrics())
            ray_metrics = self._fetch_ray_metrics()
            if ray_metrics:
                self.wfile.write(b"\n")
                self.wfile.write(ray_metrics)
        except Exception as e:
            self.wfile.write(f"# Error generating metrics: {e}\n".encode())
            logger.exception("Error generating metrics")

    def _fetch_ray_metrics(self) -> bytes:
        try:
            with urllib.request.urlopen(f"http://localhost:{self.server.ray_metrics_port}", timeout=5) as resp:
                return resp.read()
        except Exception as e:
            logger.debug(f"Failed to fetch Ray metrics from port {self.server.ray_metrics_port}: {e}")
            return b""

    def log_message(self, format, *args):
        # Suppress per-request access logging.
        pass


def start_unified_metrics_server(port: int = 9090, ray_metrics_port: int = 8080, daemon: bool = False) -> HTTPServer:
    """Start the aggregating /metrics server in a background thread and return it.

    With ``daemon=False`` (the sidecar default) the serving thread keeps the process alive;
    pass ``daemon=True`` when embedding into a process that should exit on its own.
    """
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        logger.warning("PROMETHEUS_MULTIPROC_DIR is not set — multiprocess (vLLM) metrics will be empty.")

    server = HTTPServer(("0.0.0.0", port), UnifiedMetricsHandler)
    server.ray_metrics_port = ray_metrics_port
    thread = threading.Thread(name="UnifiedMetricsServer", target=server.serve_forever, daemon=daemon)
    thread.start()
    logger.info(f"Unified metrics server started on port {port} (ray metrics port {ray_metrics_port})")
    return server


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=9090, help="port to serve the unified /metrics on")
    parser.add_argument("--ray-metrics-port", type=int, default=8080, help="Ray's --metrics-export-port")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    start_unified_metrics_server(port=args.port, ray_metrics_port=args.ray_metrics_port, daemon=False)


if __name__ == "__main__":
    main()
