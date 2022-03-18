import numpy as np
import onnxruntime as ort

from pathlib import PosixPath

from pizza_challenge.pipelines.training import ClassifierDataLoader, to_numpy


class OnnxModel:
    def __init__(self, model_path: PosixPath):
        """
        Initialize the ONNX model.

        Parameters
        ----------
        model_path: PosixPath
            Path to the ONNX model file.
        """
        self.model_path = model_path
        self.model = ort.InferenceSession(str(model_path))
        self.processor = ClassifierDataLoader()
        self.labels = ["False", "True"]

    
    def predict(self, sample: str) -> bool:
        """
        Predict the label of the given text sample.

        Parameters
        ----------
        sample: str
            Text sample to predict.
        """
        tokens = self.processor.tokenize_data(sample)
        ort_inputs = {
            "input_ids": to_numpy(tokens["input_ids"]),
            "attention_mask": to_numpy(tokens["attention_mask"]),
        }
        ort_outputs = self.model.run(None, ort_inputs)
        return self.labels[np.argmax(ort_outputs[0])]
