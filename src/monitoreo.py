"""Registro de predicciones y comparación contra los datos de entrenamiento.

No guarda el texto de los comentarios, solo clase, confianza y longitud, para no almacenar
información identificable de los estudiantes.
"""
import json
import logging
import os
import threading
import time
from pathlib import Path

from src.modelo import DEFAULT_VERSION, ROOT, cargar_modelo

MIN_MUESTRA = 30          # con menos predicciones las diferencias son solo ruido
UMBRAL_PROPORCION = 0.15  # diferencia absoluta en la proporción de una clase
UMBRAL_LONGITUD = 0.5     # diferencia relativa en la longitud media
_lock = threading.Lock()


def _ruta() -> Path:
    # shortcut: archivo local que se pierde si el servidor no tiene disco persistente, al desplegar usar un volumen
    return Path(os.getenv("MONITOREO_RUTA", ROOT / "monitoreo" / "predicciones.jsonl"))


def registrar(resultados, version: str = DEFAULT_VERSION) -> None:
    """Agrega las predicciones válidas al registro. Un fallo de disco no debe tumbar la predicción."""
    filas = [json.dumps({"t": round(time.time()), "version": version, "polaridad": r["polaridad"],
                         "confianza": r["confianza"], "longitud": len(r["texto"])})
             for r in resultados if r["valido"]]
    if not filas:
        return
    try:
        ruta = _ruta()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with _lock, ruta.open("a", encoding="utf-8") as f:
            f.write("\n".join(filas) + "\n")
    except OSError:
        logging.exception("No se pudo escribir el registro de monitoreo")


def resumen(ultimas: int = 1000, version: str = DEFAULT_VERSION) -> dict:
    """Compara las últimas predicciones con la referencia de entrenamiento y devuelve alertas."""
    _, meta = cargar_modelo(version)
    ref = meta["referencia"]
    ruta = _ruta()
    filas = [json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines()[-ultimas:]] if ruta.exists() else []
    n = len(filas)
    salida = {"n": n, "referencia": ref, "alertas": []}
    if n == 0:
        return salida
    proporciones = {c: round(sum(f["polaridad"] == c for f in filas) / n, 4) for c in ref["proporcion_clases"]}
    longitud = round(sum(f["longitud"] for f in filas) / n, 1)
    salida.update(proporcion_clases=proporciones, longitud_media=longitud,
                  confianza_media=round(sum(f["confianza"] for f in filas) / n, 4),
                  proporcion_baja_confianza=round(sum(f["confianza"] < 0.5 for f in filas) / n, 4))
    if n < MIN_MUESTRA:
        return salida
    for c, p in proporciones.items():
        if abs(p - ref["proporcion_clases"][c]) > UMBRAL_PROPORCION:
            salida["alertas"].append(f"La clase {c} está en {p:.2f} y en entrenamiento era {ref['proporcion_clases'][c]:.2f}")
    if abs(longitud - ref["longitud_media"]) / ref["longitud_media"] > UMBRAL_LONGITUD:
        salida["alertas"].append(f"La longitud media es {longitud:.0f} y en entrenamiento era {ref['longitud_media']:.0f}")
    return salida
