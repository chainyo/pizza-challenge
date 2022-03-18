import json
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
import torch

from pytorch_lightning import (
    Trainer, 
    callbacks,
    seed_everything,
)
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from typing import  Any, Dict

from .model import RequestClassifier, ClassifierDataLoader


def prepare_data(path: str) -> Dict[str, list]:
    """
    Prepare the data for training.

    Parameters
    ----------
    path: str
        Path to the file containing the data.
    
    Returns
    -------
    Dict[list]
        Dictionary containing the training, validation and test data as lists.
    """
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], 
        df["requester_received_pizza"], 
        test_size=0.2, random_state=3, 
        stratify=df["requester_received_pizza"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=3, stratify=y_train
    )
    return {
        "X_train": X_train.tolist(),
        "X_val": X_val.tolist(),
        "X_test": X_test.tolist(),
        "y_train": y_train.tolist(),
        "y_val": y_val.tolist(),
        "y_test": y_test.tolist(),
    }


def training_loop(
    dataset: Dict[str, list],
    parameters: Dict[str, Any],
    dir_path: str,
) -> Dict[str, bool]:
    """
    Training loop of the Classification model.

    Parameters
    ----------
    dataset: Dict[list]
        Dictionary containing the training, validation and test data.
    parameters: Dict[str, Any]
        Dictionary containing the parameters for training.

    Returns
    -------
    Dict[str, bool]
        Returns True if training is successful.
    """
    seed_everything(42, workers=True)
    logger = WandbLogger(project=parameters["wandb_project"])
    gpu_value = 1 if torch.cuda.is_available() else 0 # Check if GPU is available

    model = RequestClassifier(parameters["n_classes"])
    data_module = ClassifierDataLoader(dataset, parameters["batch_size"])

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=dir_path,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping_callback = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=parameters["max_epochs"],
        progress_bar_refresh_rate=10, 
        gpus=gpu_value, 
        logger=logger, 
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=False,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    return {
        "model_path": trainer.logger.experiment.dirpath,
        "data_module": data_module,
    }


def convert_model_to_onnx(
    training_outputs: Dict[str, Any],
    export_path: str,
) -> Dict[str, Any]:
    """
    Convert the trained model to ONNX format.

    Parameters
    ----------
    training_outputs: Dict[str, Any]
        Dictionary containing the trained model path and data module.
    export_path: str
        Path to the directory where the model will be exported.
    
    Returns
    -------
    Dict[str, Any]
        Return onnx model path and input sample for validation.
    """
    model_path = training_outputs["model_path"]
    data_module = training_outputs["data_module"]

    model = RequestClassifier.load_from_checkpoint(model_path)
    input_batch = next(iter(data_module.test_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }

    onnx_model_path = f"{export_path}/model.onnx"
    torch.onnx.export(
        model,
        (input_sample["input_ids"], input_sample["attention_mask"]),
        onnx_model_path,
        export_params=True,
        opset_version=11,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    )
    return {
        "pytorch_model_path": model_path,
        "onnx_model_path": onnx_model_path,
        "input_sample": input_sample,
    }


def validate_model(conversion_outputs: Dict[str, Any]) -> Dict[str, bool]:
    """
    Compare the same computed output of the PyTorch model with the one exported in ONNX format.
    
    Parameters
    ----------
    conversion_outputs: Dict[str, Any]
        Dictionary containing the trained model path, exported model path and input sample.

    Raises
    ------
    ValueError
        If the computed output of the PyTorch model is different from the exported ONNX model.

    Returns
    -------
    Dict[str, bool]
        Return True if the output of the exported model is the same as the output of the PyTorch model.
    """
    onnx_model_path = conversion_outputs["onnx_model_path"]
    onnx_model = onnx.load(onnx_model_path)
    try:
        torch.onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
            raise ValueError(f"ONNX model is not valid: {e}")
    
    try:
        input_sample = conversion_outputs["input_sample"]
        pytorch_model_path = conversion_outputs["pytorch_model_path"]
        pytorch_model = RequestClassifier.load_from_checkpoint(pytorch_model_path)
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(**input_sample)
        ort_session = ort.InferenceSession(onnx_model_path)
        ort_inputs = {
            "input_ids": np.expand_dims(input_sample["input_ids"], axis=0),
            "attention_mask": np.expand_dims(input_sample["attention_mask"], axis=0),
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(
            to_numpy(pytorch_output), ort_outputs["output"][0], rtol=1e-03, atol=1e-05
        )
        print("🎉 ONNX model is valid. 🎉")
    except Exception as e:
        raise ValueError(f"ONNX model is not valid: {e}")
    return {"validation_success": True}


def to_numpy(tensor: torch.Tensor):
    """
    Converts a tensor to numpy array.

    Parameters
    ----------
    tensor: torch.Tensor
        Tensor to be converted.
    
    Returns
    -------
    numpy.ndarray
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
