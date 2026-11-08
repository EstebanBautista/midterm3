"""Pruebas básicas del modelo guardado y de la API. Ejecución: .venv/Scripts/python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from src.modelo import predecir  # noqa: E402
from src.preprocesamiento import limpiar_texto  # noqa: E402


def test_limpieza():
    assert limpiar_texto("NO explica 😡 parcial 2") == "no no_explica emo_neg parcial num"
    assert "sin_embargo" in limpiar_texto("Sin embargo, bien")


def test_prediccion_valida():
    r = predecir(["El curso fue excelente, el profesor muy claro y dedicado."])[0]
    assert r["valido"] and r["polaridad"] in {"negativo", "neutral", "positivo"}
    assert abs(sum(r["probabilidades"].values()) - 1) < 1e-3


def test_entradas_invalidas():
    r = predecir(["", "   ", None, "x" * 6000])
    assert all(not x["valido"] for x in r)


def test_api():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json={"texto": "Los parciales fueron injustos y nada claros."})
        assert r.status_code == 200 and "probabilidades" in r.json()
        assert c.post("/predict", json={"texto": ""}).status_code == 422
        b = c.post("/predict/batch", json={"textos": ["Muy buen curso", ""]}).json()
        assert b["n"] == 2 and b["resultados"][1]["valido"] is False
