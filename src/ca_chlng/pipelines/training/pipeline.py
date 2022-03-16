from multiprocessing.spawn import prepare
from kedro.pipeline import Pipeline, node

from .nodes import prepare_data, training_loop


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
                outputs="training_done",
                name="training_node",
            ),
        ]
    )