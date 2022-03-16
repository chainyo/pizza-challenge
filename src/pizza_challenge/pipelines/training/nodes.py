import json
import pandas as pd
import torch

from pytorch_lightning import (
    Trainer, 
    callbacks,
    seed_everything,
)
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from typing import  Any, Dict

from .model import RequestClassifier


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

    model = RequestClassifier(
        batch_size=parameters["batch_size"], 
        n_classes=parameters["n_classes"], 
        dataset=dataset
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=parameters["dir_path"],
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

    trainer.fit(model)
    trainer.test(model)
    return {"training_done": True}
