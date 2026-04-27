# YOLOv8 FastAPI Deployment

A simple production-ready structure for deploying a YOLOv8 model with FastAPI.

## Project layout

- `.gitignore` - ignore local/runtime files
- `.dockerignore` - ignore unnecessary files during Docker build
- `Dockerfile` - container image definition
- `docker-compose.yml` - local service composition
- `requirements.txt` - Python dependencies
- `models/model.pt` - placeholder for your trained YOLOv8 weights
- `src/` - inference and model loader code
- `tests/` - basic API tests

## Setup

```bash
cd ml-project
python3 -m venv DEV
source DEV/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn src.inference:app --reload --host 0.0.0.0 --port 8000
```

## Test the API

```bash
curl -X POST "http://127.0.0.1:8000/detect" -F "file=@/path/to/image.jpg"
```

## Docker

Build and run:

```bash
docker compose up --build
```

## Model file

Place your trained model into `models/model.pt` or update the `MODEL_PATH` environment variable.
