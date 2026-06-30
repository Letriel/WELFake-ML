"""ONNX Runtime wrapper around model.onnx (the exported LSTM classifier)."""
import numpy as np
import onnxruntime as ort

from .config import FAKE_IS_HIGH, MODEL_PATH, THRESHOLD


class FakeNewsModel:
    def __init__(self, model_path=MODEL_PATH):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        # Resolve input name dynamically (robust to the 'keras_tensor' export name).
        self.input_name = self.session.get_inputs()[0].name

    def predict_proba(self, padded: np.ndarray) -> np.ndarray:
        """padded: float32 (N, maxlen) -> sigmoid probabilities (N,)."""
        out = self.session.run(None, {self.input_name: padded})[0]
        return np.asarray(out).reshape(-1)

    @staticmethod
    def classify(prob):
        """Map a sigmoid output to (label, confidence, probability)."""
        prob = float(prob)
        if FAKE_IS_HIGH:
            label = "FAKE" if prob >= THRESHOLD else "REAL"
        else:
            label = "REAL" if prob >= THRESHOLD else "FAKE"
        confidence = prob if prob >= 0.5 else 1.0 - prob
        return label, confidence, prob
