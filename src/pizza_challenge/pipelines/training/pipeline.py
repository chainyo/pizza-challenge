from kedro.pipeline import Pipeline, node

from .nodes import (
    convert_model_to_onnx,
    prepare_data, 
    training_loop,
    validate_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=prepare_data,
                inputs=[
                    "params:training_data_path",
                ],
                outputs="dataset",
                name="preparing_data_node",
            ),
            node(
                func=training_loop,
                inputs=[
                    "dataset",
                    "params:training",
                ],
                outputs="training_outputs",
                name="training_node",
            ),
            node(
                func=convert_model_to_onnx,
                inputs=[
                    "training_outputs",
                    "params:export_path",
                ],
                outputs="conversion_outputs",
                name="conversion_node",
            ),
            node(
                func=validate_model,
                inputs=[
                    "conversion_outputs",
                ],
                outputs="validation_done",
                name="validation_node",
            ),
        ]
    )
