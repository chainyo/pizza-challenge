"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from pizza_challenge.pipelines import training


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    training_pipeline = training.create_pipeline()
    return {
        "__default__": training_pipeline,
        "training": training_pipeline,
    }
