from kedro.pipeline import node, Pipeline
from benchmark_prophet.pipelines.cross_validation.nodes import load_data, preprocess_data, run_cv


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_data,
                inputs=["preprocessed_time_series", "parameters"],
                outputs='dataset',
                name="load_data",
            ),
            node(
                func=run_cv,
                inputs=["dataset", "parameters"],
                outputs='results_cv_with_predictions_refactored',
                name="run_cv",
            ),
        ]
    )
