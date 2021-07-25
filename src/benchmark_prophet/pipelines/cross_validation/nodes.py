import pandas as pd
import numpy as np
from benchmark_prophet.pipelines.utils import (
    _split_df,
    _crossvalidation_split_df,
    _create_train,
    preprocess_data_cv,
    _train_predict_np,
    _train_predict_sklearn,
    _train_predict_arima,
    _train_predict_sarima,
    _train_predict_prophet,
    _train_predict_lstm,
    _train_predict_nbeats,
    _train_predict_deepar,
    _train_predict_tft
)
from ray import tune
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import re


def load_data(preprocessed_time_series, params):

    particular_time_series = params["input"].split('_')[-1].isnumeric()
    if particular_time_series:
        pattern = re.compile(fr'^{params["input"]}')
    else:
        pattern = re.compile(fr'^{params["input"]}_\d')

    time_series_list = [
        i
        for i in preprocessed_time_series.keys()
        if pattern.match(i)
    ]
    dataset = {}
    for ts in time_series_list:
        one_time_series = preprocessed_time_series[ts]().drop_duplicates()
        one_time_series["ds"] = pd.to_datetime(one_time_series["ds"])
        dataset.update({ts: one_time_series})
    return dataset


def model_cv_run(tr, vl, model_parameters_list, config):
    model_parameters = {
        k: config[k] for k in model_parameters_list if k in list(config.keys())
    }
    if config["method"].lower() == "np":
        (y_rolled, y_pred_rolled), additional_info = _train_predict_np(
            tr, vl, model_parameters, config["freq"], config["test_size"]
        )
        return (y_rolled, y_pred_rolled), additional_info

    elif config["method"].lower() == 'lstm':
        (y_rolled, y_pred_rolled), additional_info = _train_predict_lstm(
            tr, vl, model_parameters, config["freq"], config["test_size"]
        )
        return (y_rolled, y_pred_rolled), additional_info

    elif config["method"].lower() == 'nbeats':
        (y_rolled, y_pred_rolled), additional_info = _train_predict_nbeats(
            tr, vl, model_parameters, config["freq"], config["test_size"]
        )
        return (y_rolled, y_pred_rolled), additional_info

    elif config["method"].lower() == 'deepar':
        (y_rolled, y_pred_rolled), additional_info = _train_predict_deepar(
            tr, vl, model_parameters, config["freq"], config["test_size"]
        )
        return (y_rolled, y_pred_rolled), additional_info

    elif config["method"].lower() == 'tft':
        (y_rolled, y_pred_rolled), additional_info = _train_predict_tft(
            tr, vl, model_parameters, config["freq"], config["test_size"]
        )
        return (y_rolled, y_pred_rolled), additional_info

    elif config["method"].lower() == "prophet":
        n_lags = model_parameters.pop("n_lags")
        n_forecasts = model_parameters.pop("n_forecasts")
        (y_rolled, y_pred_rolled), additional_info = _train_predict_prophet(
            tr, vl, model_parameters, config["test_size"], config["freq"]
        )
        return (y_rolled, y_pred_rolled), additional_info

    elif config["method"].lower() in ["rf", "gb", "mlp"]:
        model_parameters.pop("n_lags")
        n_forecasts = model_parameters.pop("n_forecasts")
        if config["method"].lower() == "rf":
            reg = RandomForestRegressor(**model_parameters)
        elif config["method"].lower() == "gb":
            reg = GradientBoostingRegressor(**model_parameters)
        elif config["method"].lower() == "mlp":
            reg = MLPRegressor(**model_parameters)
        (y_rolled, y_pred_rolled), additional_info = _train_predict_sklearn(
            tr, vl, reg, n_forecasts
        )
        return (y_rolled, y_pred_rolled), additional_info
    elif config["method"].lower() == "arima":
        n_lags = model_parameters.pop("n_lags")
        n_forecasts = model_parameters.pop("n_forecasts")
        ma = model_parameters.pop("ma")
        integration = model_parameters.pop("i")
        order = (n_lags, integration, ma)
        model_parameters.update({"order": order})
        (y_rolled, y_pred_rolled), additional_info = _train_predict_arima(
            tr, vl, model_parameters, config["test_size"], n_forecasts
        )
        return (y_rolled, y_pred_rolled), additional_info
    elif config["method"].lower() == "sarima":
        freq = config["freq"]
        if "seasonal" in config.keys():
            seasonal = config["seasonal"]
        else:
            if freq == "H":
                seasonal = 24
            elif freq == "30min":
                seasonal = 48
            elif freq == "10min":
                seasonal = 6 * 24
            elif freq == "D":
                seasonal = 7

        n_lags = model_parameters.pop("n_lags")
        n_forecasts = model_parameters.pop("n_forecasts")
        ma = model_parameters.pop("ma")
        integration = model_parameters.pop("i")
        order = (n_lags, integration, ma)
        seasonal_order = (1, 0, 0, seasonal)
        model_parameters.update({"order": order})
        model_parameters.update({"seasonal_order": seasonal_order})

        (y_rolled, y_pred_rolled), additional_info = _train_predict_sarima(
            tr, vl, model_parameters, config["test_size"], n_forecasts
        )
        if "seasonal" not in config.keys():
            additional_info.update({"seasonal": seasonal})

        return (y_rolled, y_pred_rolled), additional_info


def run_and_process_results(cv_dataset, model_parameters_list, config):
    y_true = []
    y_pred = []
    for fold, (tr, vl) in enumerate(cv_dataset):
        (y_rolled, y_pred_rolled), additional_info = model_cv_run(
            tr, vl, model_parameters_list, config
        )

        y_true_df = pd.DataFrame(
            np.vstack(y_rolled),
            columns=[f"true_value_{i}" for i in range(1, config["n_forecasts"] + 1)],
        )
        y_true_df["fold"] = fold
        y_pred_df = pd.DataFrame(
            np.vstack(y_pred_rolled),
            columns=[f"forecast_{i}" for i in range(1, config["n_forecasts"] + 1)],
        )
        y_pred_df["fold"] = fold
        y_true.append(y_true_df)
        y_pred.append(y_pred_df)

        if fold == 0:
            config.update(additional_info)
        else:
            for key in additional_info.keys():
                try:
                    config[key].append(additional_info[key][0])
                except:
                    continue
    y_true = pd.concat(y_true)
    y_pred = pd.concat(y_pred)

    cols = y_true.columns
    y_true = y_true.reset_index()
    y_true.columns = ["step"] + list(cols)

    cols = y_pred.columns
    y_pred = y_pred.reset_index()
    y_pred.columns = ["step"] + list(cols)

    return y_true, y_pred


def run_cv(dataset, params):
    use_exact_values = int(params["use_exact_values"])

    method = params["method"]

    if params["method"].lower() == "np":
        parameter_list = 'NPParameterList'
    elif params["method"].lower() == 'lstm':
        parameter_list = 'LSTMParameterList'
    elif params["method"].lower() == 'nbeats':
        parameter_list = 'NBeatsParameterList'
    elif params["method"].lower() == 'deepar':
        parameter_list = 'DeepARParameterList'
    elif params["method"].lower() == 'tft':
        parameter_list = 'TFTParameterList'
    elif params["method"].lower() == "prophet":
        parameter_list = 'ProphetParameterList'
    elif params["method"].lower() in ["rf", "gb", "mlp"]:
        parameter_list = 'SklearnParameterList'
    elif params["method"].lower() == "arima":
        parameter_list = 'ArimaParameterList'
    elif params["method"].lower() == "sarima":
        parameter_list = 'SarimaParameterList'

    model_parameters_list = params[parameter_list]
    model_parameters = {
        k: params[k] for k in model_parameters_list if k in list(params.keys())
    }
    if method.lower() == "prophet":
        model_parameters.update({"n_lags": list([0])})
        model_parameters.update({"n_forecasts": list([1])})
    time_series_list = tune.grid_search(list(dataset.keys()))

    variable_params = {
        k: tune.grid_search(model_parameters[k]) for k in model_parameters.keys()
    }
    variable_params.update({"time_series": time_series_list})

    def train(config):
        config.update({"method": method})
        ts = dataset[config["time_series"]]
        dataset_name = "_".join(config["time_series"].split("_")[:-1])
        freq = params[dataset_name]["freq"]
        cv_dataset, test_size, train_folds = preprocess_data_cv(
            ts, params, config["n_lags"], config["n_forecasts"]
        )

        config.update({"freq": freq, "test_size": test_size})

        y_true, y_pred = run_and_process_results(
            cv_dataset, model_parameters_list, config
        )
        train_folds = _create_train(config, train_folds)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "train_folds": train_folds
        }

    analysis = tune.run(
        train,
        config=variable_params,
        log_to_file=False,
        verbose=True,
        raise_on_failed_trial=True,
        checkpoint_freq=0,
        resources_per_trial={"cpu": 6, "gpu": 0},
    )


    results = analysis.results_df[
        [col for col in analysis.results_df.columns if "config." in col]
        + ["y_true", "y_pred"]
    ].reset_index(drop=True)

    train_folds_from_results = analysis.results_df['train_folds'].reset_index(drop=True)
    train_fold = []
    for row in range(len(train_folds_from_results)):
        train_fold.append(train_folds_from_results.iloc[row])
    train_folds_from_results = pd.concat(train_fold)

    config_columns = [col for col in results.columns if ("config" in col)]

    dfs = []
    for row in results.iterrows():
        r = row[1]
        y_true = r["y_true"]
        y_pred = r["y_pred"]
        df = y_true.merge(y_pred)
        for col in config_columns:
            if type(r[col]) == list:
                df[col] = "_".join([str(i) for i in r[col]])
            else:
                df[col] = r[col]

        dfs.append(df)
    results = pd.concat(dfs)

    cv_results_with_predictions = {
        f"results_cv_pred_{params['input']}_{method}": results
    }

    train_fold_results = {f"train_fold_{params['input']}_{method}":train_folds_from_results}

    return cv_results_with_predictions, train_fold_results
