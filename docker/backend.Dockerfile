# Backend: FastAPI + the ML pipeline (CPU). Build from the project folder:
#   docker compose build
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# GDAL / PROJ libraries for rasterio, geopandas and pyproj
RUN apt-get update && apt-get install -y --no-install-recommends gdal-bin libgdal-dev libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only PyTorch keeps the image ~3 GB smaller than the CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r requirements.txt -r backend/requirements.txt

COPY ml /app/ml
COPY backend /app/backend

WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
