# ...existing code...
from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()

REQUESTS_TOTAL = Counter(
    "app_http_requests_total", "Total de requisições HTTP", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "app_http_request_latency_seconds", "Latência de requisição HTTP (s)", ["path"]
)
IN_PROGRESS_REQUESTS = Gauge("app_http_requests_in_progress", "Requisições em andamento")
REQUEST_ERRORS = Counter("app_http_request_errors_total", "Total de requisições com erro", ["method", "path"])

@router.get("/", include_in_schema=False)
@router.get("/metrics", include_in_schema=False)
def metrics():
    """Expose Prometheus metrics."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)