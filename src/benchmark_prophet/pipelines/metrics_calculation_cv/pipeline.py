from kedro.pipeline import node, Pipeline
from benchmark_prophet.pipelines.metrics_calculation_cv.nodes import calculate_metrics


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=calculate_metrics,
                inputs=["results_cv_with_predictions_refactored", "train_fold_results", "parameters"],
                outputs='metrics_df_cv',
                name="metrics_from_cv_calculation",
            )
        ]
    )
