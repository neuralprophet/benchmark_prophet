from kedro.pipeline import node, Pipeline
from benchmark_prophet.pipelines.testing.nodes import load_data, run_testing


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
                func=run_testing,
                inputs=["dataset", "parameters"],
                outputs=['results_test_with_predictions_refactored', 'train_fold_results_test'],
                name="run_testing",
            ),
        ]
    )
