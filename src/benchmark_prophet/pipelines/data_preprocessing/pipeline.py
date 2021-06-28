from kedro.pipeline import node, Pipeline
from benchmark_prophet.pipelines.data_preprocessing.nodes import preprocess_into_dataframe


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_into_dataframe,
                inputs=["time_series_input", "parameters"],
                outputs="preprocessed_time_series",
                name="preprocessing_time_series",
            ),
        ]
    )
