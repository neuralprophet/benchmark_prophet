from kedro.pipeline import node, Pipeline
from benchmark_prophet.pipelines.metrics_calculation_test.nodes import calculate_metrics


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=calculate_metrics,
                inputs=["results_test_with_predictions_refactored", "train_fold_results_test", "parameters"],
                outputs='metrics_df_test',
                name="metrics_from_test_calculation",
            )
        ]
    )
