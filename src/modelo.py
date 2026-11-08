"""Carga el pipeline entrenado una sola vez y expone una única función de predicción.

La usan tanto el servicio FastAPI como la app Streamlit, para que ambos se comporten igual.
Solo se deben cargar archivos .joblib de confianza: la deserialización puede ejecutar código arbitrario.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "modelos"
DEFAULT_VERSION = "v1"
MAX_CHARS = 5000


@lru_cache(maxsize=4)
def cargar_modelo(version: str = DEFAULT_VERSION):
    """Devuelve (pipeline, metadatos) de una versión del modelo. Se carga una sola vez por proceso."""
    import joblib

    ruta = MODEL_DIR / f"modelo_polaridad_{version}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el modelo {ruta.name}")
    meta = json.loads((MODEL_DIR / f"modelo_polaridad_{version}.json").read_text(encoding="utf-8"))
    return joblib.load(ruta), meta


def _probabilidades(modelo, textos):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(textos)
    # LinearSVC no tiene predict_proba: se usa softmax sobre los puntajes de decisión
    s = modelo.decision_function(textos)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predecir(textos, version: str = DEFAULT_VERSION):
    """Clasifica una lista de comentarios.

    Devuelve un diccionario por comentario: {texto, polaridad, confianza, probabilidades, valido, error}.
    Los comentarios vacíos o demasiado largos no pasan por el modelo y se marcan con valido=False.
    """
    modelo, _ = cargar_modelo(version)
    resultados = [None] * len(textos)
    validos = []
    for i, t in enumerate(textos):
        t = "" if t is None else str(t)
        if not t.strip():
            resultados[i] = {"texto": t, "polaridad": None, "confianza": None, "probabilidades": None,
                             "valido": False, "error": "Comentario vacío"}
        elif len(t) > MAX_CHARS:
            resultados[i] = {"texto": t[:80] + "…", "polaridad": None, "confianza": None, "probabilidades": None,
                             "valido": False, "error": f"Comentario supera {MAX_CHARS} caracteres"}
        else:
            validos.append((i, t))
    if validos:
        idx, txt = zip(*validos)
        probs = _probabilidades(modelo, list(txt))
        clases = list(modelo.classes_)
        for i, t, p in zip(idx, txt, probs):
            k = int(np.argmax(p))
            resultados[i] = {"texto": t, "polaridad": clases[k], "confianza": round(float(p[k]), 4),
                             "probabilidades": {c: round(float(v), 4) for c, v in zip(clases, p)},
                             "valido": True, "error": None}
    return resultados
