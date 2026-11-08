# Clasificación de comentarios de evaluación docente (SET)

- **Curso:** Sistemas Inteligentes, Pontificia Universidad Javeriana Cali
- **Profesor:** Jefferson Amado Peña Torres
- **Integrantes:** Esteban David Muñoz Bautista, Nicolás Carreño Tascón, Brayan Zuluaga, Juan Manuel Canchala Giménez

Este proyecto clasifica comentarios abiertos de estudiantes sobre la docencia como **negativo**, **neutral** o **positivo**.

- **Midterm 3:** construye y evalúa el modelo (notebook) y lo reporta en un artículo con la plantilla de la revista *Ingeniería y Universidad*.
- **Proyecto final:** integra el modelo en un producto mínimo viable (MVP), con un servicio FastAPI y una aplicación Streamlit.

## Resultados (conjunto de prueba, n = 124)

| Modelo | macro-F1 CV | macro-F1 prueba | Exactitud prueba |
|---|---|---|---|
| Dummy (clase más frecuente) | 0,197 | 0,197 | 0,419 |
| Mejor TF-IDF en prueba (Linear SVM) | 0,643 | 0,661 | 0,669 |
| **Híbrido TF-IDF + embeddings, Logistic Regression (seleccionado)** | **0,711** | **0,714** | **0,718** |

El modelo se seleccionó por su macro-F1 en validación cruzada, no por su resultado en prueba.

## Estructura del repositorio

```
Dataset/                 CSV originales (617 comentarios útiles tras la limpieza; el nombre del docente evaluado viene reemplazado por la etiqueta PROFESOR)
notebooks/               midterm3_clasificacion_SET.ipynb (pasos 1-7)
build_notebook.py        genera el notebook
src/preprocesamiento.py  limpieza de texto + transformador de embeddings (compartido por notebook, API y app)
src/modelo.py            carga el modelo una vez, valida la entrada y devuelve clase + probabilidades
modelos/                 modelo_polaridad_v1.joblib + .json (métricas, parámetros, versiones)
api/main.py              servicio FastAPI
app/streamlit_app.py     interfaz Streamlit
articulo/                artículo (.docx y .pdf) y su generador
figuras/                 figuras en alta resolución y tablas de resultados
plantilla/               plantilla oficial y guía de postulación de la revista
tests/                   pruebas básicas
```

## Instalación (Windows)

Requiere Python 3.12. No necesita Docker.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

La primera ejecución descarga de Hugging Face el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` (unos 470 MB).

## Cómo ejecutar

Todos los comandos se ejecutan desde la raíz del repositorio.

| Tarea | Comando |
|---|---|
| Re-entrenar (ejecutar el notebook) | `.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace notebooks\midterm3_clasificacion_SET.ipynb` |
| Regenerar el artículo | `.venv\Scripts\python articulo\build_articulo.py` |
| API | `.venv\Scripts\uvicorn api.main:app --reload`, luego abrir http://127.0.0.1:8000/docs |
| App | `.venv\Scripts\streamlit run app\streamlit_app.py` |
| Pruebas | `.venv\Scripts\python -m pytest -q` |

### Ejemplo de uso de la API

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d "{\"texto\": \"El profesor explica muy bien pero los parciales fueron injustos\"}"
```

La respuesta incluye `polaridad`, `confianza`, `probabilidades` (una por clase) y `tiempo_ms`. Un comentario vacío devuelve HTTP 422.

## Gestión del modelo

- **Se guarda el pipeline completo, no solo el clasificador.** El archivo `.joblib` contiene la limpieza, el vectorizador y el clasificador juntos, así el preprocesamiento es idéntico en entrenamiento y producción.
- **El modelo no se re-entrena al iniciar.** La API y la app cargan el `.joblib` una sola vez.
- **Solo se cargan archivos `.joblib` propios.** La deserialización puede ejecutar código arbitrario, por eso no se cargan modelos de origen desconocido.
- **Mismas versiones en entrenamiento y producción.** Las versiones de `scikit-learn` y sus dependencias están fijadas en `requirements.txt`. El `.json` de cada modelo registra las versiones con que se entrenó.
- **Versionado y reversión.** Cada modelo se guarda como `modelo_polaridad_vN.joblib` + `.json`.
  - **Actualización:** se entrena, se guarda una nueva versión `vN+1` y se cambia `DEFAULT_VERSION` en `src/modelo.py`.
  - **Reversión:** se vuelve a poner en `DEFAULT_VERSION` la versión anterior.

## Uso de LLM

Se utilizaron LLM como apoyo en la exploración de datos, el preprocesamiento y el análisis de errores; las decisiones fueron revisadas por los integrantes.
