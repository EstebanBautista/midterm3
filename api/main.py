"""Servicio de predicción para comentarios de evaluación docente.

Ejecución desde la raíz del repositorio:
    .venv/Scripts/uvicorn api.main:app --reload
Documentación interactiva: http://127.0.0.1:8000/docs
"""
import os
import secrets
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modelo import DEFAULT_VERSION, cargar_modelo, predecir  # noqa: E402
from src.monitoreo import registrar, resumen  # noqa: E402

_clave_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_peticiones = defaultdict(deque)


def proteger(request: Request, clave: str | None = Depends(_clave_header)):
    """Exige la clave si API_KEY está definida y limita las peticiones por minuto de cada cliente."""
    esperada = os.getenv("API_KEY")
    if esperada and not (clave and secrets.compare_digest(clave.encode(), esperada.encode())):
        raise HTTPException(status_code=401, detail="Clave de API inválida o ausente")
    # shortcut: contador en memoria de un solo proceso y por IP directa, con varios workers o un proxy usar Redis o el límite del proxy
    limite = int(os.getenv("RATE_LIMIT_POR_MINUTO", "60"))
    ahora = time.monotonic()
    ventana = _peticiones[request.client.host]
    while ventana and ahora - ventana[0] >= 60:
        ventana.popleft()
    if len(ventana) >= limite:
        raise HTTPException(status_code=429, detail="Demasiadas peticiones, intente en un minuto")
    ventana.append(ahora)


@asynccontextmanager
async def lifespan(_app):
    # el modelo se carga una sola vez al iniciar; aquí nunca se re-entrena
    cargar_modelo(DEFAULT_VERSION)
    predecir(["calentamiento"])
    yield


app = FastAPI(title="SET - Clasificador de polaridad", version="1.0.0", lifespan=lifespan,
              description="Clasifica comentarios de evaluación docente en negativo, neutral o positivo.")


class Comentario(BaseModel):
    texto: str = Field(..., examples=["El profesor explica muy bien y resuelve todas las dudas."])


class Lote(BaseModel):
    textos: list[str] = Field(..., max_length=1000, examples=[["Excelente curso.", "Los parciales fueron injustos."]])


@app.get("/health")
def health():
    _, meta = cargar_modelo(DEFAULT_VERSION)
    return {"status": "ok", "modelo": meta}


@app.post("/predict", dependencies=[Depends(proteger)])
def predict(c: Comentario):
    t0 = time.perf_counter()
    r = predecir([c.texto])[0]
    if not r["valido"]:
        raise HTTPException(status_code=422, detail=r["error"])
    registrar([r])
    r["tiempo_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return r


@app.post("/predict/batch", dependencies=[Depends(proteger)])
def predict_batch(lote: Lote):
    t0 = time.perf_counter()
    res = predecir(lote.textos)
    registrar(res)
    return {"n": len(res), "tiempo_ms": round(1000 * (time.perf_counter() - t0), 1), "resultados": res}


@app.get("/monitoreo", dependencies=[Depends(proteger)])
def monitoreo():
    return resumen()
