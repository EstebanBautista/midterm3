"""Load the trained pipeline once and expose a single predict function.

Shared by the FastAPI service and the Streamlit app so both behave the same.
Only load .joblib files you trust: unpickling can run arbitrary code.
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
    """Return (pipeline, metadata) for a model version. Cached: loaded once per process."""
    import joblib

    ruta = MODEL_DIR / f"modelo_polaridad_{version}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el modelo {ruta.name}")
    meta = json.loads((MODEL_DIR / f"modelo_polaridad_{version}.json").read_text(encoding="utf-8"))
    return joblib.load(ruta), meta


def _probabilidades(modelo, textos):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(textos)
    # LinearSVC has no predict_proba: softmax over decision scores as a fallback
    s = modelo.decision_function(textos)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predecir(textos, version: str = DEFAULT_VERSION):
    """Classify a list of comments.

    Returns one dict per input: {texto, polaridad, confianza, probabilidades, valido, error}.
    Empty or too-long comments are not sent to the model and come back with valido=False.
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
