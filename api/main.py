"""Servicio de predicción para comentarios de evaluación docente.

Ejecución desde la raíz del repositorio:
    .venv/Scripts/uvicorn api.main:app --reload
Documentación interactiva: http://127.0.0.1:8000/docs
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modelo import DEFAULT_VERSION, cargar_modelo, predecir  # noqa: E402


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


@app.post("/predict")
def predict(c: Comentario):
    t0 = time.perf_counter()
    r = predecir([c.texto])[0]
    if not r["valido"]:
        raise HTTPException(status_code=422, detail=r["error"])
    r["tiempo_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return r


@app.post("/predict/batch")
def predict_batch(lote: Lote):
    t0 = time.perf_counter()
    res = predecir(lote.textos)
    return {"n": len(res), "tiempo_ms": round(1000 * (time.perf_counter() - t0), 1), "resultados": res}
