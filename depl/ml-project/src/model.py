from pathlib import Path
from ultralytics import YOLO


class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None

    def load(self):
        self._validate_model_path()
        if self.model is None:
            self.model = YOLO(str(self.model_path))
        return self.model

    def _validate_model_path(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
