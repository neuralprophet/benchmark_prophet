import pandas as pd
import numpy as np
# from benchmark_prophet.pipelines.utils import

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sktime.performance_metrics.forecasting import median_absolute_percentage_error, mean_absolute_scaled_error, MeanSquaredScaledError

import re
def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

def wape(a, f):
    return np.sum(np.abs(a-f))/np.sum(a)

def calculate_metrics(results_cv_with_predictions_refactored, train_fold_results, params):
    method = params['method']

    df_pred = results_cv_with_predictions_refactored[f"results_cv_pred_{params['input']}_{method}_horizon_{params['n_forecasts']}"]().copy(deep = True)
    train_folds = train_fold_results[f"train_fold_{params['input']}_{method}_horizon_{params['n_forecasts']}"]().copy(deep = True)

    cols_config = [col for col in df_pred.columns if 'config.' in col]

    ts = df_pred['config.time_series'].unique()
    cols_config.remove('config.time_series')

    method = df_pred['config.method'].unique()[0]
    cols_config.remove('config.method')

    cols_config.remove('config.training_time')
    cols_config.remove('config.predicting_time')
    cols_config.remove('config.freq')
    cols_config.remove('config.test_size')
    cols_config.remove('config.val_size')

    try:
        cols_config.remove('config.chosen_lr')
        cols_config.remove('config.chosen_n_epoch')
    except:
        pass

    configurations = df_pred[cols_config].drop_duplicates().reset_index(drop=True)

    metrices = []
    for time_series in ts:
        df_ts = df_pred[df_pred['config.time_series'] == time_series]
        train_ts = train_folds[train_folds['config.time_series'] == time_series]
        metrices_ts = []
        for row_n, config in configurations.iterrows():
            metrics_df = pd.DataFrame()
            n_forecasts = int(config.to_dict()['config.n_forecasts'])
            part = pd.DataFrame(config.to_dict(), index=[0]).merge(df_ts, on=cols_config, how='left')
            part_train = pd.DataFrame(config.to_dict(), index=[0]).merge(train_ts, on=cols_config, how='left')
            folds = part['fold'].unique()
            metrics_df['fold'] = folds
            metrics_df['ts'] = time_series
            metrics_df['method'] = method
            for col in cols_config:
                metrics_df[col] = config.to_dict()[col]
            for fold in folds:
                part_fold = part[part['fold'] == fold]
                part_train_fold = part_train[part_train['fold'] == fold]
                part_train_fold = part_train_fold[part_train_fold['tr/vl'] == 'tr']
                y_train = part_train_fold.y.values
                for frcst in range(1, n_forecasts + 1):
                    y_true = part_fold[[f'true_value_{frcst}']].values
                    y_pred = part_fold[[f'forecast_{frcst}']].values
                    metrics_df.loc[metrics_df['fold'] == fold,
                                   f'MSE_forecast_{frcst}'] = mean_squared_error(y_true, y_pred)
                    metrics_df.loc[metrics_df['fold'] == fold,
                                   f'RMSE_forecast_{frcst}'] = np.sqrt(mean_squared_error(y_true, y_pred))
                    metrics_df.loc[metrics_df['fold'] == fold,
                                   f'MAE_forecast_{frcst}'] = mean_absolute_error(y_true, y_pred)
                    metrics_df.loc[metrics_df['fold'] == fold,
                                   f'MASE_forecast_{frcst}'] = mean_absolute_scaled_error(y_true=y_true,
                                                                                          y_pred=y_pred,
                                                                                          y_train=y_train)
                    try:
                        metrics_df.loc[metrics_df['fold'] == fold,
                                       f'MAPE_forecast_{frcst}'] = median_absolute_percentage_error(y_true=y_true,
                                                                                                    y_pred=y_pred)
                    except:
                        metrics_df.loc[metrics_df['fold'] == fold,
                                       f'MAPE_forecast_{frcst}'] = None

                    msse = MeanSquaredScaledError()
                    metrics_df.loc[metrics_df['fold'] == fold,
                                   f'MSSE_forecast_{frcst}'] = msse(y_true=y_true,
                                                                    y_pred=y_pred,
                                                                    y_train=y_train)


                    try:
                        metrics_df.loc[metrics_df['fold'] == fold,
                                       f'SMAPE_forecast_{frcst}'] = smape(y_true, y_pred)
                    except:
                        metrics_df.loc[metrics_df['fold'] == fold,
                                       f'SMAPE_forecast_{frcst}'] = None

                    try:
                        metrics_df.loc[metrics_df['fold'] == fold,
                                       f'WAPE_forecast_{frcst}'] = wape(y_true, y_pred)
                    except:
                        metrics_df.loc[metrics_df['fold'] == fold,
                                       f'WAPE_forecast_{frcst}'] = None




            metrices_ts.append(metrics_df)
        metrices.append(pd.concat(metrices_ts))

    metrics = {
        f"metrics_cv_{params['input']}_{method}_horizon_{params['n_forecasts']}": pd.concat(metrices)
    }

    return metrics


