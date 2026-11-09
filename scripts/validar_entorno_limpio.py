"""Verifica que el modelo guardado carga y predice igual en un entorno recién instalado.

Crea un entorno virtual temporal con requirements.txt, corre las pruebas, compara las predicciones
contra las del entorno actual y deja la evidencia en modelos/validacion_entorno_limpio.json.
Ejecución desde la raíz del repositorio:
    .venv/Scripts/python scripts/validar_entorno_limpio.py
"""
import json
import os
import subprocess
import sys
import tempfile
import venv
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SALIDA = ROOT / "modelos" / "validacion_entorno_limpio.json"
CASOS = ["El profesor explica muy bien y siempre resuelve dudas.",
         "Los parciales no tienen nada que ver con lo visto en clase, muy desorganizado.",
         "El curso estuvo bien, normal.",
         "Buen contenido, aunque los parciales fueron injustos.",
         ""]
SONDA = """
import json, sys
import sklearn
sys.path.insert(0, sys.argv[1])
from src.modelo import predecir
print("RESULTADO" + json.dumps({"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                                "predicciones": predecir(json.loads(sys.argv[2]))}))
"""
ENTORNO = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def sonda(python, cwd):
    p = subprocess.run([str(python), "-c", SONDA, str(ROOT), json.dumps(CASOS)], cwd=cwd, env=ENTORNO,
                       capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(next(x for x in p.stdout.splitlines() if x.startswith("RESULTADO"))[len("RESULTADO"):])


def iguales(a, b):
    if a["polaridad"] != b["polaridad"] or a["valido"] != b["valido"]:
        return False
    if not a["valido"]:
        return True
    return all(abs(a["probabilidades"][c] - b["probabilidades"][c]) < 1e-3 for c in a["probabilidades"])


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        referencia = sonda(sys.executable, tmp)
        destino = Path(tmp) / "venv"
        venv.create(destino, with_pip=True)
        python = destino / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "-q", "--disable-pip-version-check",
                        "-r", str(ROOT / "requirements.txt")], env=ENTORNO, check=True)
        limpio = sonda(python, tmp)
        pruebas = subprocess.run([str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, env=ENTORNO,
                                 capture_output=True, text=True, encoding="utf-8")
    coinciden = all(iguales(a, b) for a, b in zip(referencia["predicciones"], limpio["predicciones"]))
    evidencia = {"fecha": date.today().isoformat(), "python": limpio["python"], "scikit_learn": limpio["scikit_learn"],
                 "pruebas": pruebas.stdout.strip().splitlines()[-1], "pruebas_correctas": pruebas.returncode == 0,
                 "predicciones_coinciden": coinciden,
                 "predicciones": [{"comentario": c, "polaridad": r["polaridad"], "probabilidades": r["probabilidades"]}
                                  for c, r in zip(CASOS, limpio["predicciones"])]}
    SALIDA.write_text(json.dumps(evidencia, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidencia, indent=2, ensure_ascii=False))
    return 0 if coinciden and pruebas.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
