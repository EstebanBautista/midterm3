"""Genera el artículo del midterm sobre la plantilla oficial de la revista Ingeniería y Universidad.

Las cifras se leen de los CSV/JSON que exporta el notebook, así el artículo siempre coincide
con la última ejecución. Ejecución desde la raíz del repositorio:
    .venv/Scripts/python articulo/build_articulo.py
"""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figuras"
TEMPLATE = ROOT / "plantilla" / "plantilla_IyU.docx"
OUT = ROOT / "articulo" / "articulo_midterm3_SET.docx"

meta = json.loads((ROOT / "modelos" / "modelo_polaridad_v1.json").read_text(encoding="utf-8"))
base = pd.read_csv(FIG / "tabla_linea_base.csv")
ajuste = pd.read_csv(FIG / "tabla_ajuste.csv")
test = pd.read_csv(FIG / "tabla_test.csv")
rep = pd.read_csv(FIG / "tabla_reporte_final.csv", index_col=0)
split = pd.read_csv(FIG / "tabla_split.csv", index_col=0)
errores = pd.read_csv(FIG / "tabla_errores.csv")
n_err = int(errores["n"].sum())
n_err_neutral = int(errores[(errores["real"] == "neutral") | (errores["pred"] == "neutral")]["n"].sum())
_pn = int(errores[(errores["real"] == "positivo") & (errores["pred"] == "negativo")]["n"].sum())
n_pos_neg_txt = ("ningún comentario positivo se clasificó como negativo" if _pn == 0
                 else f"solo {_pn} comentario(s) positivo(s) se clasificaron como negativos")

f3 = lambda x: f"{x:.3f}".replace(".", ",")
pct = lambda x: f"{100 * x:.1f}".replace(".", ",") + " %"

best_cv = ajuste.iloc[0]
dummy_test = test[test["Modelo"].str.startswith("Dummy")].iloc[0]
emb_lr_test = test[(test["Representación"] == "Embeddings") & (test["Modelo"] == "Logistic Regression")].iloc[0]
tfidf_best_test = test[test["Representación"] == "TF-IDF"].sort_values("macro-F1", ascending=False).iloc[0]
base_dummy = base[base["Modelo"] == "Dummy (más frecuente)"].iloc[0]
base_best = base[~base["Modelo"].str.startswith("Dummy")].iloc[0]
base_best_tfidf = base[(base["Representación"] == "TF-IDF") & (~base["Modelo"].str.startswith("Dummy"))].iloc[0]
N_TOTAL = int(split["train"].sum() + split["test"].sum())

# ----------------------------------------------------------------- helpers
doc = Document(TEMPLATE)
body = doc.element.body
for el in list(body):
    if el.tag != qn("w:sectPr"):
        body.remove(el)

TNR = "Times New Roman"
# Afiliación de los autores (la versión para revisión ciega se envía sin nombres)
AFILIACION = "Pontificia Universidad Javeriana Cali, Sistemas Inteligentes"
CAP_FONT = "Tw Cen MT Condensed"


def _run(p, text, size=12, bold=None, italic=None, font=TNR):
    r = p.add_run(text)
    r.font.name = font
    r._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    r.font.size = Pt(size)
    r.bold, r.italic = bold, italic
    return r


def para(text="", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=None, italic=None, font=TNR, after=6, before=0):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.line_spacing = 1.0
    # marcado simple: **negrita** y ~~cursiva~~
    for i, part in enumerate(text.split("**")):
        for j, sub in enumerate(part.split("~~")):
            if sub:
                _run(p, sub, size, bold=True if i % 2 else bold, italic=True if j % 2 else italic, font=font)
    return p


def heading(text, level=2):
    p = doc.add_paragraph(text, style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    return p


def subheading(text):
    # estilo "Subtitles" de la plantilla: Tw Cen MT Condensed 14 pt, negrita cursiva, sangría 1,25 cm
    p = para(text, size=14, bold=True, italic=True, font=CAP_FONT, after=4, before=8)
    p.paragraph_format.first_line_indent = Cm(1.25)
    p.paragraph_format.keep_with_next = True
    return p


def caption(text):
    p = para(text, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font=CAP_FONT, after=3, before=8)
    p.paragraph_format.keep_with_next = True
    return p


def source(text="Fuente: elaboración propia."):
    return para(text, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font=CAP_FONT, after=10)


def figure(path, width_cm, title):
    caption(title)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    source()


def _borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for side, val in (("top", top), ("bottom", bottom)):
        if val:
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(val)); e.set(qn("w:color"), "000000")
            b.append(e)
    tcPr.append(b)


def table(title, header, rows, widths_cm, src="Fuente: elaboración propia."):
    """Tabla con tres líneas horizontales (superior, encabezado, inferior)."""
    caption(title)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths_cm[j])
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(1)
            _run(p, str(val), size=10, bold=True if i == 0 else None)
            if i == 0:
                _borders(c, top=8, bottom=6)
            elif i == len(rows):
                _borders(c, bottom=8)
    source(src)
    return t


# ----------------------------------------------------------------- title block
para("", after=0)
para("Initial machine learning classification of polarity in open-ended student evaluation of teaching comments",
     size=30, align=WD_ALIGN_PARAGRAPH.RIGHT, bold=True, font=CAP_FONT, after=6)
para("Clasificación inicial de la polaridad en comentarios abiertos de evaluación docente mediante aprendizaje automático",
     size=20, align=WD_ALIGN_PARAGRAPH.RIGHT, font=CAP_FONT, after=14)
para("Esteban David Muñoz Bautista, Nicolás Carreño Tascón, Brayan Zuluaga, Juan Manuel Canchala Giménez",
     size=12, align=WD_ALIGN_PARAGRAPH.RIGHT, bold=True, after=2)
para(AFILIACION, size=11, align=WD_ALIGN_PARAGRAPH.RIGHT, italic=True, after=14)

# ----------------------------------------------------------------- abstract
heading("Abstract")
para(f"**Objective:** to evaluate an initial machine learning model that classifies the polarity (negative, neutral, positive) of "
     f"open-ended student evaluation of teaching comments written in Spanish. **Materials and methods:** {N_TOTAL} comments from "
     f"seven computer science courses were cleaned and their labels normalized. A stratified 80/20 split was used. TF-IDF "
     f"representations, multilingual sentence embeddings and a hybrid of both were compared with Logistic Regression, linear SVM, "
     f"Naive Bayes and Random Forest against a dummy baseline, using 5-fold stratified cross-validation and grid search optimized "
     f"for macro-F1. **Results and discussion:** semantic embeddings outperformed TF-IDF in every classifier. The selected model "
     f"(hybrid representation with Logistic Regression) reached a cross-validated macro-F1 of {best_cv['macro-F1 CV']:.3f} and a "
     f"test macro-F1 of {meta['macro_f1_test']:.3f} (accuracy {meta['accuracy_test']:.3f}), against {dummy_test['macro-F1']:.3f} for "
     f"the baseline. Negative comments were recognized best; most errors involved the neutral class and mixed-polarity comments. "
     f"**Conclusions:** a lightweight, reproducible pipeline is viable as the core of a minimum viable product, although label "
     f"quality and the small sample limit performance. Future work includes label validation, aspect-based "
     f"classification and deployment as an MVP.")
para("**Keywords:** sentiment analysis; student evaluation of teaching; text classification; sentence embeddings; machine learning.")

heading("Resumen")
para(f"**Objetivo:** evaluar un modelo inicial de aprendizaje automático que clasifique la polaridad (negativa, neutral, positiva) "
     f"de comentarios abiertos de evaluación docente escritos en español. **Materiales y métodos:** se depuraron {N_TOTAL} "
     f"comentarios de siete cursos de ingeniería y se normalizaron sus etiquetas. Se usó una partición estratificada 80/20. Se "
     f"compararon TF-IDF, embeddings multilingües y su combinación con Regresión Logística, "
     f"SVM lineal, Naive Bayes y Random Forest frente a una línea base trivial, con validación cruzada estratificada "
     f"y búsqueda en malla por macro-F1. **Resultados y discusión:** los embeddings superaron a TF-IDF en "
     f"todos los clasificadores. El modelo seleccionado (representación híbrida con Regresión Logística) obtuvo un macro-F1 de "
     f"{f3(best_cv['macro-F1 CV'])} en validación cruzada y de {f3(meta['macro_f1_test'])} en prueba (exactitud "
     f"{f3(meta['accuracy_test'])}), frente a {f3(dummy_test['macro-F1'])} de la línea base. La clase negativa fue la mejor "
     f"reconocida; la mayoría de errores involucró la clase neutral. **Conclusiones:** un "
     f"pipeline liviano y reproducible es viable como núcleo de un producto mínimo viable, aunque la calidad del etiquetado y el "
     f"tamaño de la muestra limitan el desempeño. Se recomienda validar las etiquetas, abordar la "
     f"clasificación por aspectos y desplegar el modelo en un MVP.")
para("**Palabras clave:** análisis de sentimientos; evaluación docente; clasificación de texto; embeddings de oraciones; aprendizaje automático.")

# ----------------------------------------------------------------- introduction
heading("Introducción")
para("Las encuestas de evaluación docente, conocidas como SET por sus siglas en inglés, combinan preguntas cerradas con comentarios abiertos. "
     "En esos comentarios está la información más detallada sobre la experiencia del estudiante, pero revisarlos a mano de forma sistemática "
     "resulta inviable por su volumen, y el análisis de sentimientos permite resumirlos de manera automática [1].")
para("Esta área pasó de usar léxicos de opinión y clasificadores convencionales sobre bolsas de palabras [2], [3] a representaciones "
     "neuronales basadas en transformers [4], [5] y modelos de lenguaje de gran tamaño, o LLM [6]. Los estudios de mapeo sistemático muestran un "
     "crecimiento sostenido de estas técnicas en la retroalimentación estudiantil [7], aunque su adopción práctica sigue siendo limitada, ya que "
     "hay pocas implementaciones, su uso en flujos institucionales es bajo y los resultados son poco accionables.")
para("Este trabajo es la primera fase de un proyecto que busca integrar un clasificador en un producto mínimo viable (MVP). En esta fase se "
     "busca responder qué combinación de representación de texto y clasificador da el mejor macro-F1 al clasificar la polaridad de comentarios de "
     "evaluación docente en español con un conjunto de datos pequeño. Para esto se comparan representaciones dispersas basadas en frecuencia de "
     "término y frecuencia inversa de documento (TF-IDF) con embeddings de oraciones, usando cuatro clasificadores clásicos y un protocolo "
     "reproducible que evita la fuga de información.")

# ----------------------------------------------------------------- methods
heading("Materiales y métodos")
subheading("Conjunto de datos y exploración")
para("Se usaron tres archivos CSV suministrados por el curso, con las columnas id, curso, feedback, polaridad_basica, grano_fino, "
     "emocion, aspectos y semestre. Los comentarios vienen de siete asignaturas como Sistemas Operativos, Bases de Datos y Sistemas Inteligentes, "
     "de los semestres 2023-1 a 2024-2. En la exploración inicial, apoyada por un LLM, Claude Opus 5.5 con nivel de esfuerzo medio, para revisar el esquema, se encontraron los problemas de "
     "calidad que resume la Tabla 1, mientras que no hubo valores faltantes, comentarios vacíos ni duplicados.")
table("Tabla 1. Problemas de calidad encontrados y tratamiento aplicado.",
      ["Problema", "Tratamiento"],
      [["Fila de encabezado repetida en cada archivo (4 filas)", "Eliminada"],
       ["polaridad_basica con 9 valores distintos (negativo, ['negativo'], [\"['positivo']\"], positiva...)", "Normalizada a 3 clases"],
       ["Etiqueta inválida ['posicion'] (1 fila)", "Descartada"],
       ["grano_fino y emocion casi constantes (99 % con valor 3 y neutral)", "No usadas como objetivo"],
       ["Nombre de curso con error tipográfico (opertativos)", "Corregido"],
       ["Sin identificador de autor", "Documentado como limitación"]],
      [10.5, 5.0])
para(f"El conjunto final tiene {N_TOTAL} comentarios, de los cuales {int(split.loc['neutral', ['train', 'test']].sum())} son neutrales, "
     f"{int(split.loc['negativo', ['train', 'test']].sum())} negativos y {int(split.loc['positivo', ['train', 'test']].sum())} positivos, como muestra la Figura 1. La longitud media "
     f"es de 45 palabras y es parecida entre clases, así que la longitud por sí sola no indica la polaridad.")
figure(FIG / "fig1_eda.png", 16.0, "Figura 1. Distribución de clases, polaridad por curso y longitud de los comentarios.")

subheading("Definición de la tarea y preprocesamiento")
para("La tarea es una clasificación multiclase en la que cada comentario se asigna a negativo, neutral o positivo. La métrica principal es el "
     "macro-F1, definido en la ecuación 1, que le da el mismo peso a las tres clases y por eso es adecuado cuando están desbalanceadas [8]. El "
     "preprocesamiento va integrado en el pipeline para que se aplique igual en entrenamiento y en inferencia, y consiste en pasar el texto a "
     "minúsculas, eliminar tildes, reemplazar URL, correos y números por marcadores como una anonimización básica, convertir emojis en los "
     "marcadores emo_pos y emo_neg y marcar las negaciones, de modo que no claro queda como no no_claro. Las palabras vacías no se eliminan porque "
     "negaciones e intensificadores como no, muy, nada o poco cambian la polaridad. El LLM apoyó la definición de estas reglas, y las decisiones "
     "finales fueron validadas por el equipo.")
eq = para("", align=WD_ALIGN_PARAGRAPH.CENTER, after=6, before=2)
for txt, it, sub in [("macro-F1 = (F1", False, False), ("1", False, True), (" + F1", False, False), ("2", False, True),
                     (" + F1", False, False), ("3", False, True), (") / 3,   F1", False, False), ("k", True, True), (" = 2 ", False, False),
                     ("P", True, False), ("k", True, True), (" ", False, False), ("R", True, False), ("k", True, True), (" / (", False, False),
                     ("P", True, False), ("k", True, True), (" + ", False, False), ("R", True, False), ("k", True, True),
                     (")          (1)", False, False)]:
    _run(eq, txt, 12, italic=it).font.subscript = sub or None
para("donde ~~P~~k y ~~R~~k son la precisión y la exhaustividad (recall) de la clase ~~k~~.", after=6)

subheading("División de los datos")
para(f"Se hizo una partición estratificada con {int(split['train'].sum())} comentarios de entrenamiento, el 80 %, y "
     f"{int(split['test'].sum())} de prueba, el 20 %, con semilla fija igual a 42. Ambas particiones conservan la proporción de clases, que es "
     f"cerca de 33 % negativo, 42 % neutral y 25 % positivo, y el conjunto de prueba solo se usó en la evaluación final. Como los datos no "
     f"incluyen un identificador de estudiante no fue posible agrupar por autor, por lo que se verificó que no existieran textos duplicados "
     f"entre archivos para reducir el riesgo de fuga de información [9].")

subheading("Representación vectorial del texto")
para("Se compararon tres representaciones. La primera es TF-IDF [10], que une n-gramas de palabras de 1 a 2 y de caracteres de 3 a 5 con "
     "escala sublineal, estos últimos aportan robustez frente a errores ortográficos. La segunda son embeddings semánticos obtenidos con el modelo "
     "preentrenado paraphrase-multilingual-MiniLM-L12-v2 de Hugging Face [11]-[13], que genera vectores normalizados de 384 dimensiones y, como no "
     "se ajusta con los datos del curso, no introduce fuga. La tercera es una representación híbrida que concatena TF-IDF y embeddings. El "
     "vectorizador TF-IDF se ajustó solo con los datos de entrenamiento de cada partición de validación.")

subheading("Entrenamiento inicial, ajuste de hiperparámetros y evaluación")
para("Como referencia se usaron dos DummyClassifier, uno de clase más frecuente y otro estratificado. Se compararon Regresión Logística, "
     "máquinas de vectores de soporte (SVM) lineales, Naive Bayes, multinomial para TF-IDF y gaussiano para embeddings, y Random Forest, con pesos "
     "de clase balanceados cuando el clasificador lo permite. Todos se implementaron como Pipeline de scikit-learn [14] y se evaluaron con "
     "validación cruzada estratificada de 5 particiones sobre el entrenamiento. Luego los cuatro clasificadores con TF-IDF, los tres mejores con "
     "embeddings y Regresión Logística con la representación híbrida se ajustaron con búsqueda en malla (GridSearchCV) usando macro-F1 como "
     "criterio, como se ve en la Tabla 3, mientras que la búsqueda aleatoria [15] queda como alternativa para espacios más grandes. El modelo "
     "final se eligió por su macro-F1 en validación cruzada y no en prueba, para no sesgar la selección. Sobre la prueba se reportan exactitud, "
     "precisión, exhaustividad o recall, macro-F1 y matriz de confusión. El LLM se usó como apoyo en el análisis de errores. Los comentarios "
     "venían anonimizados, con el nombre del docente evaluado reemplazado por la etiqueta PROFESOR.")

# ----------------------------------------------------------------- results
heading("Resultados")
subheading("Línea base")
base_tab = base[~(base["Modelo"].str.startswith("Dummy") & (base["Representación"] == "Embeddings"))].copy()
base_tab.loc[base_tab["Modelo"].str.startswith("Dummy"), "Representación"] = "N/A"
rows = [[r["Modelo"], r["Representación"], f3(r["macro-F1"]), f3(r["Desv. est."]), f3(r["Accuracy"])] for _, r in base_tab.iterrows()]
table("Tabla 2. Línea base: validación cruzada estratificada de 5 particiones sobre el entrenamiento.",
      ["Modelo", "Representación", "macro-F1", "Desv. est.", "Exactitud"], rows, [5.0, 3.2, 2.4, 2.4, 2.4])
para(f"Todos los modelos superaron al Dummy de clase más frecuente, que obtuvo un macro-F1 de {f3(base_dummy['macro-F1'])}, y los embeddings "
     f"superaron a TF-IDF con todos los clasificadores. El mejor modelo inicial fue {base_best['Modelo']} con embeddings, con un macro-F1 de "
     f"{f3(base_best['macro-F1'])}, frente a {f3(base_best_tfidf['macro-F1'])} del mejor modelo con TF-IDF, que fue {base_best_tfidf['Modelo']}. Naive Bayes "
     f"multinomial tuvo el peor desempeño, con un recall macro de solo "
     f"{f3(base[(base['Representación'] == 'TF-IDF') & (base['Modelo'] == 'Naive Bayes')].iloc[0]['Recall'])}, como muestra la Figura 2.")
figure(FIG / "fig2_linea_base.png", 13.0, "Figura 2. Comparación de la línea base por modelo y representación (macro-F1).")

subheading("Ajuste de hiperparámetros")


def fmt_params(s):
    s = s.replace("clf__", "").replace("vec__pal__", "").replace("'", "")
    return s.strip("{}")


rows = [[f"{r['Modelo']} + {r['Representación']}", f3(r["macro-F1 CV"]), f3(r["Desv. est."]), fmt_params(r["Mejores parámetros"])]
        for _, r in ajuste.iterrows()]
table("Tabla 3. Resultados de GridSearchCV (macro-F1 en validación cruzada) y mejores hiperparámetros.",
      ["Modelo + representación", "macro-F1", "Desv. est.", "Mejores hiperparámetros"], rows, [5.4, 2.0, 2.0, 6.3])
C_best = fmt_params(best_cv["Mejores parámetros"]).replace("C: ", "C = ")
para(f"El ajuste igualó o mejoró la línea base en todos los modelos ajustados. La representación híbrida con Regresión Logística ({C_best}) obtuvo "
     f"el mayor macro-F1 en validación cruzada, {f3(best_cv['macro-F1 CV'])}, y fue seleccionada como modelo final. Las diferencias entre los tres "
     f"mejores modelos están dentro de una desviación estándar, así que deben interpretarse con cautela.")

subheading("Evaluación final en el conjunto de prueba")
rows = [[f"{r['Modelo']}" + ("" if r["Representación"] == "N/A" else f" + {r['Representación']}"), f3(r["Accuracy"]),
         f3(r["Precision (macro)"]), f3(r["Recall (macro)"]), f3(r["macro-F1"])] for _, r in test.iterrows()]
table(f"Tabla 4. Desempeño en el conjunto de prueba (n = {int(split['test'].sum())}).",
      ["Modelo", "Exactitud", "Precisión", "Recall", "macro-F1"], rows, [6.2, 2.3, 2.3, 2.3, 2.3])
para(f"El modelo seleccionado alcanzó un macro-F1 de {f3(meta['macro_f1_test'])} y una exactitud de {f3(meta['accuracy_test'])} en prueba, "
     f"valores consistentes con la validación cruzada y sin señales de sobreajuste. En prueba algunos modelos que usan solo embeddings obtuvieron "
     f"valores algo mayores, hasta {f3(emb_lr_test['macro-F1'])} con Regresión Logística, pero esa diferencia equivale a pocas predicciones en un "
     f"conjunto de {int(split['test'].sum())} comentarios y elegir el modelo por su resultado en prueba sesgaría la estimación. Todos los modelos "
     f"con embeddings superan en al menos "
     f"{int(100 * (test[test['Representación'].isin(['Embeddings', 'Híbrido'])]['macro-F1'].min() - tfidf_best_test['macro-F1']))} puntos "
     f"al mejor modelo con TF-IDF, que obtuvo {f3(tfidf_best_test['macro-F1'])}.")
rows = [[c, f3(rep.loc[c, "precision"]), f3(rep.loc[c, "recall"]), f3(rep.loc[c, "f1-score"]), int(rep.loc[c, "support"])]
        for c in ["negativo", "neutral", "positivo"]]
rows.append(["Promedio macro", f3(rep.loc["macro avg", "precision"]), f3(rep.loc["macro avg", "recall"]),
             f3(rep.loc["macro avg", "f1-score"]), int(rep.loc["macro avg", "support"])])
table("Tabla 5. Métricas por clase del modelo seleccionado en prueba.",
      ["Clase", "Precisión", "Recall", "F1", "Soporte"], rows, [4.0, 2.6, 2.6, 2.6, 2.6])
figure(FIG / "fig3_matriz_confusion.png", 16.0, "Figura 3. Matriz de confusión del modelo seleccionado en el conjunto de prueba.")

subheading("Análisis de errores")
para(f"La clase negativa fue la mejor reconocida, con un recall de {f3(rep.loc['negativo', 'recall'])}, mientras que las clases neutral y positiva "
     f"concentraron los errores, como se observa en la Figura 3. De los {n_err} errores, {n_err_neutral} involucraron la clase neutral, y "
     f"{n_pos_neg_txt}. La revisión cualitativa, apoyada por el LLM, identificó tres categorías de error.")
para("La primera es la polaridad mixta, en comentarios que combinan elogios y críticas con conectores como pero, aunque o sin embargo, por ejemplo "
     "\"el material fue muy completo... sin embargo, algunas veces se complicaron los ejemplos\", etiquetado positivo y predicho neutral. La segunda "
     "es la frontera difusa entre neutral y positivo, con comentarios moderadamente favorables etiquetados como neutrales, por ejemplo \"el horario "
     "me resultó conveniente...\". La tercera es el ruido de etiquetado, ya que algunos comentarios tienen etiquetas poco coherentes con su contenido, "
     "como un comentario claramente positivo etiquetado como negativo, y algunos textos no son opiniones de estudiantes, lo que fija un techo al "
     "desempeño alcanzable.")
para("Los coeficientes del modelo TF-IDF con Regresión Logística muestran señales coherentes, con palabras como excelente, satisfecho, gratificante "
     "y muy para la clase positiva, no, difícil, la falta y desafiante para la negativa, y neutral, aunque, algunas y en cuanto para la neutral.")

# ----------------------------------------------------------------- conclusions
heading("Conclusiones")
para(f"Se evaluó un modelo inicial para clasificar la polaridad de comentarios de evaluación docente en español. Con {N_TOTAL} comentarios, la "
     f"representación híbrida de TF-IDF y embeddings multilingües preentrenados con un clasificador lineal alcanzó un macro-F1 de "
     f"{f3(meta['macro_f1_test'])} en prueba, más del triple que la línea base. Los embeddings semánticos superaron de forma consistente a TF-IDF, lo "
     f"que muestra el valor de reutilizar modelos preentrenados cuando hay pocos datos etiquetados.")
para("Las principales limitaciones son el tamaño reducido del conjunto de prueba, la falta de un identificador de autor y la calidad irregular de "
     "las etiquetas. Como trabajo futuro se propone revisar y validar manualmente las etiquetas con una guía de anotación, apoyada por LLM y con "
     "validación humana, extender la tarea a la clasificación por aspectos e integrar el pipeline, ya serializado con joblib, en un MVP con una "
     "API en FastAPI y una interfaz en Streamlit que permita analizar comentarios individuales o archivos CSV completos y así cerrar la brecha de "
     "accionabilidad.")

# ----------------------------------------------------------------- references
heading("Referencias")
REFS = ['B. Liu, ~~Sentiment Analysis and Opinion Mining~~. San Rafael, CA, USA: Morgan & Claypool, 2012.', 'B. Pang, L. Lee, and S. Vaithyanathan, "Thumbs up? Sentiment classification using machine learning techniques," in ~~Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)~~, 2002, pp. 79-86.', 'K. Dhanalakshmi, D. Babu, D. Rajalakshmi, and K. Rajalakshmi, "Opinion mining from student feedback data using supervised learning algorithms," in ~~Proc. 3rd MEC Int. Conf. Big Data Smart City~~, 2016, pp. 1-5.', 'A. Vaswani et al., "Attention is all you need," in ~~Adv. Neural Inf. Process. Syst. (NeurIPS)~~, vol. 30, 2017, pp. 5998-6008.', 'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in ~~Proc. NAACL-HLT~~, 2019, pp. 4171-4186.', 'T. Brown et al., "Language models are few-shot learners," in ~~Adv. Neural Inf. Process. Syst. (NeurIPS)~~, vol. 33, 2020, pp. 1877-1901.', 'Z. Kastrati, F. Dalipi, A. S. Imran, K. Pireva Nuci, and M. A. Wani, "Sentiment analysis of students\' feedback with NLP and deep learning: A systematic mapping study," ~~Appl. Sci.~~, vol. 11, no. 9, Art. no. 3986, 2021.', 'M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," ~~Inf. Process. Manage.~~, vol. 45, no. 4, pp. 427-437, 2009.', 'S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," ~~ACM Trans. Knowl. Discov. Data~~, vol. 6, no. 4, pp. 1-21, 2012.', 'G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," ~~Inf. Process. Manage.~~, vol. 24, no. 5, pp. 513-523, 1988.', 'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in ~~Proc. EMNLP-IJCNLP~~, 2019, pp. 3982-3992.', 'N. Reimers and I. Gurevych, "Making monolingual sentence embeddings multilingual using knowledge distillation," in ~~Proc. EMNLP~~, 2020, pp. 4512-4525.', 'W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," in ~~Adv. Neural Inf. Process. Syst. (NeurIPS)~~, vol. 33, 2020, pp. 5776-5788.', 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," ~~J. Mach. Learn. Res.~~, vol. 12, pp. 2825-2830, 2011.', 'J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," ~~J. Mach. Learn. Res.~~, vol. 13, pp. 281-305, 2012.']
for i, r in enumerate(REFS, 1):
    p = para(f"[{i}]\t{r}", size=10, after=3)
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.9)

doc.save(OUT)
words = sum(len(p.text.split()) for p in doc.paragraphs) + sum(len(c.text.split()) for t in doc.tables for row in t.rows for c in row.cells)
print("saved", OUT, "| palabras aprox.", words)
