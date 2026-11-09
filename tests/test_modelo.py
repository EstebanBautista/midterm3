"""Pruebas básicas del modelo guardado y de la API. Ejecución: .venv/Scripts/python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import _peticiones, app  # noqa: E402
from src.modelo import predecir  # noqa: E402
from src.monitoreo import MIN_MUESTRA, registrar, resumen  # noqa: E402
from src.preprocesamiento import limpiar_texto  # noqa: E402


@pytest.fixture(autouse=True)
def entorno_aislado(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITOREO_RUTA", str(tmp_path / "predicciones.jsonl"))
    monkeypatch.delenv("API_KEY", raising=False)
    _peticiones.clear()


def test_limpieza():
    assert limpiar_texto("NO explica 😡 parcial 2") == "no no_explica emo_neg parcial num"
    assert "sin_embargo" in limpiar_texto("Sin embargo, bien")


def test_prediccion_valida():
    r = predecir(["El curso fue excelente, el profesor muy claro y dedicado."])[0]
    assert r["valido"] and r["polaridad"] in {"negativo", "neutral", "positivo"}
    assert abs(sum(r["probabilidades"].values()) - 1) < 1e-3


def test_entradas_invalidas():
    r = predecir(["", "   ", None, float("nan"), "x" * 6000])
    assert all(not x["valido"] for x in r)


def test_api():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json={"texto": "Los parciales fueron injustos y nada claros."})
        assert r.status_code == 200 and "probabilidades" in r.json()
        assert c.post("/predict", json={"texto": ""}).status_code == 422
        b = c.post("/predict/batch", json={"textos": ["Muy buen curso", ""]}).json()
        assert b["n"] == 2 and b["resultados"][1]["valido"] is False


def test_clave_de_api(monkeypatch):
    monkeypatch.setenv("API_KEY", "clave-de-prueba")
    with TestClient(app) as c:
        cuerpo = {"texto": "Muy buen curso"}
        assert c.post("/predict", json=cuerpo).status_code == 401
        assert c.post("/predict", json=cuerpo, headers={"X-API-Key": "otra"}).status_code == 401
        assert c.post("/predict", json=cuerpo, headers={"X-API-Key": "clave-de-prueba"}).status_code == 200
        assert c.get("/monitoreo").status_code == 401
        assert c.get("/health").status_code == 200


def test_limite_de_peticiones(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_POR_MINUTO", "2")
    with TestClient(app) as c:
        codigos = [c.post("/predict", json={"texto": "Muy buen curso"}).status_code for _ in range(3)]
    assert codigos == [200, 200, 429]


def test_monitoreo_detecta_cambio_y_no_guarda_texto(tmp_path):
    secreto = "comentario con datos personales"
    registrar([{"texto": secreto, "polaridad": "negativo", "confianza": 0.9, "valido": True}] * MIN_MUESTRA)
    registrar([{"texto": "", "polaridad": None, "confianza": None, "valido": False}])
    r = resumen()
    assert r["n"] == MIN_MUESTRA and r["proporcion_clases"]["negativo"] == 1
    assert any("negativo" in a for a in r["alertas"])
    assert secreto not in (tmp_path / "predicciones.jsonl").read_text(encoding="utf-8")


def test_monitoreo_sin_alertas_con_poca_muestra():
    registrar([{"texto": "x", "polaridad": "negativo", "confianza": 0.9, "valido": True}] * (MIN_MUESTRA - 1))
    assert resumen()["alertas"] == []
