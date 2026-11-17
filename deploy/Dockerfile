# Imagen para Hugging Face Spaces. La misma imagen sirve la app o la API según SERVICIO.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# torch de CPU primero, para no descargar la versión con GPU que pesa mucho más
COPY --chown=user deploy/requirements.txt requirements-deploy.txt
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user src src
COPY --chown=user api api
COPY --chown=user app app
COPY --chown=user modelos modelos

# descarga el modelo de embeddings al construir, así el primer arranque no espera
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu')"

ARG SERVICIO=app
ENV SERVICIO=$SERVICIO
CMD if [ "$SERVICIO" = "api" ]; then \
      uvicorn api.main:app --host 0.0.0.0 --port 7860; \
    else \
      streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true \
        --server.enableXsrfProtection false --server.enableCORS false; \
    fi
