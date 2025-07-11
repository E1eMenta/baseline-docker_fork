from src.model import TruFor, preprocess_image
import numpy as np
from PIL import Image
import subprocess
import os


class TruForInference:
    def __init__(self):
        def download_and_extract_weights():
            weights_url = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
            subprocess.run(["wget", weights_url], check=True)
            subprocess.run(["unzip", "-q", "-n", "TruFor_weights.zip"], check=True)
            subprocess.run(["rm", "TruFor_weights.zip"], check=True)

        if not os.path.exists("TruFor_weights"):
            download_and_extract_weights()

        self.model = TruFor()

    def infer(self, image_path):
        img = np.array(Image.open(image_path).convert("RGB"))

        img = preprocess_image(img)
        score = self.model.detect(img)
        score = float(score)
        return score
