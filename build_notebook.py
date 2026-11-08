"""Genera notebooks/midterm3_clasificacion_SET.ipynb (pasos 1-7 de la guía del midterm)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Midterm 3 — Clasificación de comentarios de Evaluación Docente (SET)
**Sistemas Inteligentes** · Profesor: Jefferson Amado Peña Torres\n\n**Integrantes:** Esteban David Muñoz Bautista, Nicolás Carreño Tascón, Brayan Zuluaga, Juan Manuel Canchala Giménez

Objetivo: evaluar un modelo inicial de *Machine Learning* que clasifique la **polaridad** (`negativo`, `neutral`, `positivo`) de comentarios abiertos de estudiantes.
El notebook sigue los 7 pasos de la guía: (1) carga y exploración, (2) preprocesamiento, (3) división de datos, (4) representación vectorial,
(5) línea base, (6) ajuste de hiperparámetros y (7) evaluación final.
""")

code(r"""
import ast, json, re, sys, time, unicodedata, warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
ROOT = Path.cwd() if (Path.cwd() / "Dataset").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from src.preprocesamiento import LimpiadorTexto, EmbeddingTransformer, limpiar_texto, embed, EMB_NAME

SEED = 42
np.random.seed(SEED)
DATA_DIR = ROOT / "Dataset"
FIG_DIR = ROOT / "figuras"; FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR = ROOT / "modelos"; MODEL_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
print("Python", sys.version.split()[0], "| scikit-learn", sklearn.__version__, "| pandas", pd.__version__)
""")

md("## 1. Carga y exploración del CSV")
code(r"""
archivos = sorted(DATA_DIR.glob("feedback_curso_universitario*.csv"))
partes = []
for f in archivos:
    d = pd.read_csv(f, encoding="utf-8-sig")
    n0 = len(d)
    d = d[d["id"].astype(str) != "id"].copy()          # fila de encabezado repetida
    d["fuente"] = f.stem
    print(f"{f.name}: {n0} filas leídas, {n0 - len(d)} encabezado(s) duplicado(s) eliminados -> {len(d)}")
    partes.append(d)
raw = pd.concat(partes, ignore_index=True)
print("\nTotal:", raw.shape)
raw.head()
""")

code(r"""
print("Columnas:", list(raw.columns))
print("\nFaltantes por columna:\n", raw.isna().sum())
print("\nComentarios vacíos:", (raw["feedback"].fillna("").str.strip() == "").sum())
print("Comentarios duplicados (texto exacto, normalizado a minúsculas):",
      raw["feedback"].str.strip().str.lower().duplicated().sum())
for c in ["polaridad_basica", "grano_fino", "emocion", "curso", "semestre"]:
    print(f"\n--- {c} ---"); print(raw[c].value_counts(dropna=False).to_string())
""")

md(r"""
**Hallazgos de calidad**
- Cada archivo trae la fila de encabezado repetida (eliminada arriba).
- `polaridad_basica` mezcla formatos: `negativo`, `['negativo']`, `["['positivo']"]`, `positiva` y un valor inválido `['posicion']`.
- `grano_fino` y `emocion` son casi constantes (≈ 3 y `neutral`) → no sirven como variable objetivo.
- `aspectos` es una lista JSON multi-etiqueta (se deja para el proyecto final).
- No existe columna de autor/estudiante; ver paso 3.
""")

code(r"""
def normalizar_polaridad(v):
    s = str(v).strip()
    for _ in range(3):                      # desanidar "['x']" y "[\"['x']\"]"
        try:
            p = ast.literal_eval(s)
            s = str(p[0] if isinstance(p, (list, tuple)) and p else p).strip()
        except Exception:
            break
    s = s.lower()
    return {"positiva": "positivo", "negativa": "negativo"}.get(s, s)

df = raw.copy()
df["polaridad"] = df["polaridad_basica"].map(normalizar_polaridad)
invalidas = ~df["polaridad"].isin(["negativo", "neutral", "positivo"])
print("Etiquetas inválidas descartadas:", df.loc[invalidas, "polaridad_basica"].tolist())
df = df[~invalidas].reset_index(drop=True)
df["curso"] = df["curso"].str.replace("opertativos", "operativos").str.strip()
df["n_palabras"] = df["feedback"].str.split().str.len()

def parse_aspectos(v):
    try: return [a["aspecto"].strip().lower() for a in json.loads(v)]
    except Exception: return []
df["aspectos_lista"] = df["aspectos"].map(parse_aspectos)

print("Dataset final:", df.shape)
print(df["polaridad"].value_counts().to_string())
""")

code(r"""
ORDEN = ["negativo", "neutral", "positivo"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1.4, 1]})
sns.countplot(data=df, x="polaridad", order=ORDEN, ax=ax[0], palette="Set2")
ax[0].set_title("Distribución de polaridad"); ax[0].set_xlabel(""); ax[0].set_ylabel("Comentarios")
for p in ax[0].patches:
    ax[0].annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
pd.crosstab(df["curso"], df["polaridad"])[ORDEN].plot(kind="barh", stacked=True, ax=ax[1], color=sns.color_palette("Set2"))
ax[1].set_title("Polaridad por curso"); ax[1].set_ylabel("")
sns.boxplot(data=df, x="polaridad", y="n_palabras", order=ORDEN, ax=ax[2], palette="Set2")
ax[2].set_title("Longitud del comentario (palabras)"); ax[2].set_xlabel("")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig1_eda.png", dpi=200); plt.show()
print(df["n_palabras"].describe().round(1).to_string())
""")

code(r"""
asp = df.explode("aspectos_lista")["aspectos_lista"].dropna().value_counts()
print("Aspectos distintos:", asp.size); print(asp.head(12).to_string())
""")

md(r"""
## 2. Definición de la tarea y preprocesamiento
**Tarea:** clasificación multiclase de un solo comentario → {negativo, neutral, positivo}. Métrica principal: **macro-F1** (las clases están desbalanceadas).

**Preprocesamiento** (dentro del `Pipeline` para que se aplique igual en entrenamiento e inferencia):
- minúsculas y normalización Unicode (se eliminan tildes para unificar variantes con/sin tilde),
- URLs, correos y números → tokens genéricos (anonimización básica),
- emojis → tokens `emo_pos` / `emo_neg`,
- **no** se eliminan negaciones ni intensificadores (`no`, `muy`, `nada`, `poco`, `bastante`), porque cambian la polaridad; además se marca la palabra siguiente a una negación (`no claro` → `no_claro`).
""")

code(r"""
# limpiar_texto y LimpiadorTexto viven en src/preprocesamiento.py (reutilizables por la API/app)
for s in df["feedback"].sample(3, random_state=SEED):
    print("ORIGINAL :", s[:160]); print("LIMPIO   :", limpiar_texto(s)[:160], "\n")
print(limpiar_texto("El profe NO explica claro 😡 y el parcial 2 fue muy difícil 👎"))
""")

md(r"""
## 3. División de datos
Partición **estratificada 80 % / 20 %** con semilla fija. El conjunto de prueba **no se usa** hasta el paso 7; toda selección de modelo se hace con validación cruzada sobre *train*.

> El CSV **no incluye identificador de autor**, por lo que no es posible agrupar por estudiante (`GroupKFold`). Se verificó que no hay comentarios duplicados entre archivos, lo que reduce el riesgo de fuga de información. Si en el proyecto final se dispone de un ID anónimo de estudiante, debe usarse `StratifiedGroupKFold`.
""")

code(r"""
X = df["feedback"].to_numpy(dtype=object)   # numpy explícito: con pyarrow, .values devuelve un ArrowStringArray
y = df["polaridad"].to_numpy(dtype=object)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
resumen = pd.DataFrame({"train": pd.Series(y_train).value_counts(), "test": pd.Series(y_test).value_counts()}).loc[ORDEN]
resumen["% train"] = (100 * resumen["train"] / resumen["train"].sum()).round(1)
resumen["% test"] = (100 * resumen["test"] / resumen["test"].sum()).round(1)
print(f"Train: {len(X_train)} | Test: {len(X_test)}"); resumen
""")

md(r"""
## 4. Representación vectorial del texto
Se comparan dos familias, **ajustadas solo con train** (dentro de cada fold de CV):
1. **TF-IDF** de palabras (1–2 gramas) + caracteres (3–5 gramas, robusto a errores ortográficos).
2. **Embeddings semánticos** de Hugging Face: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingüe, 384 dimensiones). El modelo es preentrenado y no se ajusta con nuestros datos, por lo que no hay fuga.
""")

code(r"""
# EmbeddingTransformer y embed() viven en src/preprocesamiento.py
t0 = time.time(); embed(list(X)); print(f"Embeddings calculados para {len(X)} textos en {time.time()-t0:.1f}s")

def tfidf():
    return FeatureUnion([
        ("pal", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("car", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
""")

md(r"""
## 5. Entrenamiento inicial y línea base
Validación cruzada **estratificada de 5 folds** sobre train. Se comparan `DummyClassifier` (referencia), Logistic Regression, Linear SVM, Naive Bayes y Random Forest con ambas representaciones.
(Naive Bayes: `MultinomialNB` para TF-IDF y `GaussianNB` para embeddings, que tienen valores negativos.)
""")

code(r"""
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"macro_f1": "f1_macro", "accuracy": "accuracy", "precision": "precision_macro", "recall": "recall_macro"}

def p_tfidf(clf):  return Pipeline([("limpiar", LimpiadorTexto()), ("vec", tfidf()), ("clf", clf)])
def p_emb(clf):    return Pipeline([("emb", EmbeddingTransformer()), ("clf", clf)])

clfs = {
    "Dummy (más frecuente)": lambda: DummyClassifier(strategy="most_frequent"),
    "Dummy (estratificado)": lambda: DummyClassifier(strategy="stratified", random_state=SEED),
    "Logistic Regression":   lambda: LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
    "Linear SVM":            lambda: LinearSVC(class_weight="balanced", random_state=SEED),
    "Naive Bayes":           None,
    "Random Forest":         lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
}
filas = []
for rep, builder in [("TF-IDF", p_tfidf), ("Embeddings", p_emb)]:
    for nombre, mk in clfs.items():
        clf = (MultinomialNB() if rep == "TF-IDF" else GaussianNB()) if mk is None else mk()
        r = cross_validate(builder(clf), X_train, y_train, cv=CV, scoring=SCORING, n_jobs=1)
        filas.append({"Representación": rep, "Modelo": nombre,
                      "macro-F1": r["test_macro_f1"].mean(), "± sd": r["test_macro_f1"].std(),
                      "Accuracy": r["test_accuracy"].mean(), "Precision": r["test_precision"].mean(),
                      "Recall": r["test_recall"].mean()})
base = pd.DataFrame(filas).sort_values("macro-F1", ascending=False).reset_index(drop=True)
base.round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=base, y="Modelo", x="macro-F1", hue="Representación", palette="Set2", ax=ax)
ax.set_title("Línea base: macro-F1 (CV 5 folds sobre train)"); ax.set_xlim(0, 1); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig2_linea_base.png", dpi=200); plt.show()
""")

md(r"""
## 6. Ajuste de hiperparámetros
`Pipeline` + `GridSearchCV` con CV estratificada (5 folds), selección por **macro-F1**. Se ajustan Logistic Regression, Linear SVM y Naive Bayes con TF-IDF; Logistic Regression, Linear SVM y Random Forest con embeddings; y Logistic Regression con una representación híbrida (TF-IDF + embeddings).
""")

code(r"""
def p_hibrido(clf):
    feats = FeatureUnion([
        ("tfidf", Pipeline([("limpiar", LimpiadorTexto()), ("vec", tfidf())])),
        ("emb", EmbeddingTransformer()),
    ])
    return Pipeline([("feats", feats), ("clf", clf)])

C_GRID = [0.01, 0.1, 0.5, 1, 2, 5, 10]
busquedas = {
    ("TF-IDF", "Logistic Regression"): (p_tfidf(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
        {"vec__pal__ngram_range": [(1, 1), (1, 2)], "vec__pal__min_df": [1, 2], "clf__C": C_GRID}),
    ("TF-IDF", "Linear SVM"): (p_tfidf(LinearSVC(class_weight="balanced", random_state=SEED)),
        {"vec__pal__ngram_range": [(1, 1), (1, 2)], "vec__pal__min_df": [1, 2], "clf__C": C_GRID}),
    ("TF-IDF", "Naive Bayes"): (p_tfidf(MultinomialNB()),
        {"vec__pal__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}),
    ("Embeddings", "Logistic Regression"): (p_emb(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
        {"clf__C": C_GRID}),
    ("Embeddings", "Linear SVM"): (p_emb(LinearSVC(class_weight="balanced", random_state=SEED)),
        {"clf__C": C_GRID}),
    ("Embeddings", "Random Forest"): (p_emb(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
        {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 10, 20], "clf__min_samples_leaf": [1, 3]}),
    ("Híbrido", "Logistic Regression"): (p_hibrido(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
        {"clf__C": C_GRID}),
}
resultados, mejores = [], {}
for (rep, nom), (pipe, grid) in busquedas.items():
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, cv=CV, scoring="f1_macro", n_jobs=1, refit=True).fit(X_train, y_train)
    i = gs.best_index_
    resultados.append({"Representación": rep, "Modelo": nom, "macro-F1 CV": gs.best_score_,
                       "± sd": gs.cv_results_["std_test_score"][i], "Combinaciones": len(gs.cv_results_["params"]),
                       "Mejores parámetros": gs.best_params_, "Tiempo (s)": round(time.time() - t0, 1)})
    mejores[(rep, nom)] = gs.best_estimator_
ajuste = pd.DataFrame(resultados).sort_values("macro-F1 CV", ascending=False).reset_index(drop=True)
pd.set_option("display.max_colwidth", 120); ajuste.round(3)
""")

md(r"""
## 7. Evaluación final y comparación
Se evalúan sobre el **conjunto de prueba (20 %)**, una sola vez, los modelos ajustados. El modelo final se elige por macro-F1 en CV (no en test) para no sesgar la selección.
""")

code(r"""
filas = []
for (rep, nom), est in mejores.items():
    p = est.predict(X_test)
    filas.append({"Representación": rep, "Modelo": nom, "Accuracy": accuracy_score(y_test, p),
                  "Precision (macro)": precision_score(y_test, p, average="macro", zero_division=0),
                  "Recall (macro)": recall_score(y_test, p, average="macro", zero_division=0),
                  "macro-F1": f1_score(y_test, p, average="macro")})
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train); pd_ = dummy.predict(X_test)
filas.append({"Representación": "—", "Modelo": "Dummy (más frecuente)", "Accuracy": accuracy_score(y_test, pd_),
              "Precision (macro)": precision_score(y_test, pd_, average="macro", zero_division=0),
              "Recall (macro)": recall_score(y_test, pd_, average="macro", zero_division=0),
              "macro-F1": f1_score(y_test, pd_, average="macro")})
test_tab = pd.DataFrame(filas).sort_values("macro-F1", ascending=False).reset_index(drop=True)
test_tab.round(3)
""")

code(r"""
mejor_key = tuple(ajuste.loc[0, ["Representación", "Modelo"]])
modelo_final = mejores[mejor_key]
y_pred = modelo_final.predict(X_test)
print("Modelo seleccionado (por CV):", mejor_key, "\nParámetros:", ajuste.loc[0, "Mejores parámetros"], "\n")
print(classification_report(y_test, y_pred, labels=ORDEN, digits=3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, norm, tit in [(ax[0], None, "Conteos"), (ax[1], "true", "Normalizada por clase real")]:
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=ORDEN, normalize=norm), display_labels=ORDEN).plot(
        ax=a, cmap="Blues", colorbar=False, values_format=".2f" if norm else "d")
    a.grid(False); a.set_title(tit); a.set_xlabel("Predicción"); a.set_ylabel("Real")
fig.suptitle(f"Matriz de confusión en test — {mejor_key[1]} + {mejor_key[0]}")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig3_matriz_confusion.png", dpi=200); plt.show()
""")

md("### Análisis de errores")
code(r"""
err = pd.DataFrame({"comentario": X_test, "real": y_test, "pred": y_pred})
err = err[err["real"] != err["pred"]]
print(f"Errores: {len(err)} de {len(y_test)} ({100*len(err)/len(y_test):.1f} %)\n")
pares = err.groupby(["real", "pred"]).size().sort_values(ascending=False)
pares.rename("n").reset_index().to_csv(FIG_DIR / "tabla_errores.csv", index=False)
print(pares.to_string(), "\n")
for _, r in err.head(10).iterrows():
    print(f"[real={r.real} | pred={r.pred}] {r.comentario[:220]}\n")
""")

md(r"""
### Palabras más influyentes (modelo interpretable TF-IDF + Logistic Regression)
Los embeddings no son interpretables palabra a palabra; para explicar qué aprende el modelo se usan los coeficientes del modelo TF-IDF ajustado.
""")
code(r"""
m_tf = mejores[("TF-IDF", "Logistic Regression")]
uni = m_tf.named_steps["vec"]; clf = m_tf.named_steps["clf"]
nombres = np.array(uni.get_feature_names_out())
es_pal = np.char.startswith(nombres.astype(str), "pal__")
for i, c in enumerate(clf.classes_):
    coef = np.where(es_pal, clf.coef_[i], -np.inf)
    top = np.argsort(coef)[-12:][::-1]
    print(f"{c:>9}: ", ", ".join(n.split("__", 1)[1] for n in nombres[top]))
""")

md(r"""
### Guardar modelo (insumo para el MVP del proyecto final)
Se guarda el `Pipeline` completo (limpieza + vectorizador + clasificador) con `joblib`, junto con metadatos de versión.
""")
code(r"""
nombre_archivo = MODEL_DIR / "modelo_polaridad_v1.joblib"
joblib.dump(modelo_final, nombre_archivo)
meta = {"version": "v1", "representacion": mejor_key[0], "modelo": mejor_key[1],
        "parametros": {k: str(v) for k, v in ajuste.loc[0, "Mejores parámetros"].items()},
        "macro_f1_cv": round(float(ajuste.loc[0, "macro-F1 CV"]), 4),
        "macro_f1_test": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "accuracy_test": round(float(accuracy_score(y_test, y_pred)), 4),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)), "clases": ORDEN,
        "dataset": [f.name for f in archivos], "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__, "embeddings": EMB_NAME if mejor_key[0] != "TF-IDF" else None}
(MODEL_DIR / "modelo_polaridad_v1.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(meta, indent=2, ensure_ascii=False))

# Exportar tablas para el artículo
base.round(3).to_csv(FIG_DIR / "tabla_linea_base.csv", index=False)
ajuste.round(3).to_csv(FIG_DIR / "tabla_ajuste.csv", index=False)
test_tab.round(3).to_csv(FIG_DIR / "tabla_test.csv", index=False)
pd.DataFrame(classification_report(y_test, y_pred, labels=ORDEN, output_dict=True)).T.round(3).to_csv(FIG_DIR / "tabla_reporte_final.csv")
resumen.to_csv(FIG_DIR / "tabla_split.csv")
""")

code(r"""
# Validación de carga y predicciones: se carga el archivo guardado mediante
# src/modelo.py (el mismo código que usan la API y la app), que además rechaza textos vacíos o demasiado largos.
from src.modelo import predecir, cargar_modelo
cargar_modelo.cache_clear()
pruebas = ["El profesor explica muy bien y siempre resuelve dudas.",
           "Los parciales no tienen nada que ver con lo visto en clase, muy desorganizado.",
           "El curso estuvo bien, normal.",                                   # ambiguo
           "Buen contenido, aunque los parciales fueron injustos.",           # polaridad mixta
           "", "   ", "x" * 6000]                                             # inválidos
for r in predecir(pruebas):
    if r["valido"]:
        print(f"{r['polaridad']:>9} ({r['confianza']:.2f}) <- {r['texto'][:70]!r}")
    else:
        print(f"{'INVÁLIDO':>9}        <- {r['texto'][:20]!r}: {r['error']}")
""")

md(r"""
## Conclusiones
- **Datos:** tras la limpieza quedan 617 comentarios (259 neutrales, 205 negativos y 153 positivos). La principal dificultad del conjunto es la calidad del etiquetado: hay etiquetas en formatos distintos y comentarios con etiquetas poco coherentes con su contenido.
- **Representación:** los embeddings semánticos multilingües superan a TF-IDF con todos los clasificadores. Con pocos datos etiquetados, reutilizar un modelo preentrenado aporta más que las representaciones basadas en frecuencia de palabras.
- **Modelo seleccionado:** Logistic Regression con la representación híbrida (TF-IDF + embeddings), elegido por macro-F1 en validación cruzada (0,711). En prueba obtiene macro-F1 = 0,714 y exactitud = 0,718, frente a 0,197 del `DummyClassifier`.
- **Errores:** la clase negativa es la mejor reconocida. Casi todos los errores involucran la clase neutral, en especial comentarios con polaridad mixta («…pero…», «…sin embargo…»).
- **Siguiente fase (proyecto final):** el `Pipeline` guardado en `modelos/` ya se usa en la API (`api/main.py`) y en la aplicación (`app/streamlit_app.py`). Queda pendiente validar las etiquetas, abordar la clasificación por aspectos y desplegar el MVP.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/midterm3_clasificacion_SET.ipynb")
print("ok")
