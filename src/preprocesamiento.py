"""Text preprocessing and feature transformers shared by the notebook, API and app.

They live in an importable module (not in the notebook) so that a Pipeline saved
with joblib can be loaded again from the API or the Streamlit app.
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
    """Normalize a comment: emojis to tokens, lowercase, strip accents,
    anonymize URLs/emails/numbers, and mark the word after a negation."""
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
    t = re.sub(r"\bsin embargo\b", "sin_embargo", t)   # connector, not a negation
    out, neg = [], False
    for w in t.split():
        out.append(f"no_{w}" if neg and w not in NEGACIONES else w)
        neg = w in NEGACIONES
    return " ".join(out)


class LimpiadorTexto(BaseEstimator, TransformerMixin):
    """Pipeline step that applies limpiar_texto to every comment."""

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
    """Return L2-normalized sentence embeddings, cached per text."""
    textos = ["" if t is None else str(t) for t in textos]
    faltan = list(dict.fromkeys(t for t in textos if (model_name, t) not in _cache))
    if faltan:
        vecs = _st(model_name).encode(faltan, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        for t, v in zip(faltan, vecs):
            _cache[(model_name, t)] = v
    return np.vstack([_cache[(model_name, t)] for t in textos])


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Pipeline step that turns raw text into Sentence-Transformers embeddings.
    The pretrained model is not fitted on our data, so there is no leakage."""

    def __init__(self, model_name=EMB_NAME):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return embed(list(X), self.model_name)
