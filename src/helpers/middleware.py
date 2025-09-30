from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time
from prometheus_client import Counter, Histogram, Gauge
from src.routers.prometheus import REQUESTS_TOTAL, REQUEST_LATENCY, IN_PROGRESS_REQUESTS, REQUEST_ERRORS

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        # Incrementa o contador e o gauge no início
        IN_PROGRESS_REQUESTS.inc()
        start_time = time.time()
        
        try:
            response: Response = await call_next(request)
        except Exception as e:
            # Captura exceções e incrementa o contador de erros
            REQUEST_ERRORS.labels(
                method=request.method,
                path=request.url.path
            ).inc()
            raise e
        finally:
            # Garante que o gauge seja decrementado
            IN_PROGRESS_REQUESTS.dec()

        # Registra a latência e o contador de requisições
        process_time = time.time() - start_time
        REQUESTS_TOTAL.labels(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(path=request.url.path).observe(process_time)
        
        return response