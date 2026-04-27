import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from .model import ModelLoader
from .utils import load_image, draw_detections

app = FastAPI(title="YOLOv8 FastAPI Inference")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pt")

model_loader = ModelLoader(MODEL_PATH)
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = model_loader.load()
    return _detector


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=415, detail="Only image files are supported")

    image_bytes = await file.read()
    try:
        image = load_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    detector = get_detector()
    results = detector(image)

    detections = []
    for result in results:
        class_names = result.names  # Get class names from model
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            class_id_int = int(class_id)
            class_name = class_names.get(class_id_int, "Unknown")
            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "class_id": class_id_int,
                "class_name": class_name,
            })

    return JSONResponse({"detections": detections})


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    """Run inference and return image with plotted detections"""
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=415, detail="Only image files are supported")

    image_bytes = await file.read()
    try:
        image = load_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    detector = get_detector()
    results = detector(image)

    detections = []
    for result in results:
        class_names = result.names
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            class_id_int = int(class_id)
            class_name = class_names.get(class_id_int, "Unknown")
            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "class_id": class_id_int,
                "class_name": class_name,
            })

    # Draw detections on image
    annotated_image = draw_detections(image, detections)

    # Convert to bytes
    img_byte_arr = io.BytesIO()
    annotated_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")
def health_check():
    return {"status": "ok"}
