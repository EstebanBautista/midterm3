"""Preprocesamiento de texto y transformadores compartidos por el notebook, la API y la app.

Están en un módulo importable (y no dentro del notebook) para que el Pipeline guardado
con joblib se pueda cargar de nuevo desde la API o desde la aplicación Streamlit.
"""
import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

EMO_POS = "😀😃😄😁😊🙂😍🥰👍👏💯✨❤🎉"
EMO_NEG = "😞😟😠😡😢😭👎💔😤😒🙁☹"
NEGACIONES = {"no", "nunca", "nada", "ni", "tampoco", "sin"}
EMB_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def limpiar_texto(t) -> str:
    """Normaliza un comentario: emojis a marcadores, minúsculas, sin tildes,
    URL/correos/números anonimizados y marcado de la palabra que sigue a una negación."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        t = ""
    t = str(t)
    t = "".join(" emo_pos " if ch in EMO_POS else " emo_neg " if ch in EMO_NEG else ch for ch in t)
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " url ", t)
    t = re.sub(r"\S+@\S+", " correo ", t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"[^a-zñ_ ]+", " ", t)
    t = re.sub(r"\bsin embargo\b", "sin_embargo", t)   # conector, no es una negación
    out, neg = [], False
    for w in t.split():
        out.append(f"no_{w}" if neg and w not in NEGACIONES else w)
        neg = w in NEGACIONES
    return " ".join(out)


class LimpiadorTexto(BaseEstimator, TransformerMixin):
    """Paso del Pipeline que aplica limpiar_texto a cada comentario."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [limpiar_texto(x) for x in X]


_modelos = {}
_cache = {}


def _st(model_name):
    if model_name not in _modelos:
        from sentence_transformers import SentenceTransformer
        _modelos[model_name] = SentenceTransformer(model_name, device="cpu")
    return _modelos[model_name]


def embed(textos, model_name=EMB_NAME):
    """Devuelve embeddings de oraciones normalizados (L2), con caché por texto."""
    textos = ["" if t is None else str(t) for t in textos]
    faltan = list(dict.fromkeys(t for t in textos if (model_name, t) not in _cache))
    if faltan:
        vecs = _st(model_name).encode(faltan, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        for t, v in zip(faltan, vecs):
            _cache[(model_name, t)] = v
    return np.vstack([_cache[(model_name, t)] for t in textos])


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Paso del Pipeline que convierte texto en embeddings de Sentence-Transformers.
    El modelo preentrenado no se ajusta con los datos del curso, por lo que no hay fuga de información."""

    def __init__(self, model_name=EMB_NAME):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return embed(list(X), self.model_name)
