from typing import Protocol
from PIL import Image
from .types import Prediction

class Classifier(Protocol):
    def predict_image(self, img: Image.Image) -> Prediction: ...
