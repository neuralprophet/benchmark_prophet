from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from neuralprophet.dataset.time_dataset import tabularize_univariate_datetime
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from neuralprophet import LSTM, NBeats, TFT, DeepAR
from fbprophet import Prophet
from time import time
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
from copy import copy


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def _create_train(config, train_folds):
    df_config = pd.DataFrame(config)
    df_config = df_config.drop(
        ["training_time", "predicting_time"], axis=1
    ).drop_duplicates()
    for col in df_config.columns:
        train_folds[f"config.{col}"] = df_config[col].values[0]
    return train_folds


def _return_prediction_from_fold(forecast, vl, n_forecasts, test_size):
    forecast = vl[-test_size:].merge(forecast)
    forecast = forecast.set_index("ds")[[f"yhat{i}" for i in range(1, n_forecasts + 1)]]

    y_pred_rolled = [
        np.array(forecast).diagonal(offset=-i)
        for i in range(test_size - n_forecasts + 1)
    ]
    y = np.array(vl[-test_size:]["y"])
    y_rolled = [y[i : i + n_forecasts] for i in range(len(y) - n_forecasts + 1)]

    return y_rolled, y_pred_rolled


def _train_predict_np(tr, vl, model_parameters, freq, test_size):
    t1 = time()
    m = NeuralProphet(**model_parameters)
    metrics_tr = m.fit(tr, freq=freq, progress_bar=False)
    t2 = time()
    training_time = t2 - t1
    future_vl = m.make_future_dataframe(vl, n_historic_predictions=test_size)
    t1 = time()
    forecast_vl = m.predict(future_vl)
    t2 = time()
    predicting_time = t2 - t1
    conf = m.config_train.__dict__
    lr = conf["learning_rate"]
    n_epochs = conf["epochs"]
    y_rolled, y_pred_rolled = _return_prediction_from_fold(
        forecast_vl, vl, model_parameters["n_forecasts"], test_size
    )
    return (
        (y_rolled, y_pred_rolled),
        {
            "lr": [lr],
            "n_epoch": [n_epochs],
            "training_time": [training_time],
            "predicting_time": [predicting_time],
        },
    )


def _train_predict_lstm(tr, vl, model_parameters, freq, test_size):
    t1 = time()
    m = LSTM(**model_parameters)
    metrics_tr = m.fit(tr, freq=freq, progress_bar=False)
    t2 = time()
    training_time = t2 - t1
    future_vl = m.make_future_dataframe(vl, n_historic_predictions=test_size)
    t1 = time()
    forecast_vl = m.predict(future_vl)
    t2 = time()
    predicting_time = t2 - t1
    conf = m.config_train.__dict__
    lr = conf["learning_rate"]
    n_epochs = conf["epochs"]
    y_rolled, y_pred_rolled = _return_prediction_from_fold(
        forecast_vl, vl, model_parameters["n_forecasts"], test_size
    )
    return (
        (y_rolled, y_pred_rolled),
        {
            "lr": [lr],
            "n_epoch": [n_epochs],
            "training_time": [training_time],
            "predicting_time": [predicting_time],
        },
    )


def _train_predict_nbeats(tr, vl, model_parameters, freq, test_size):
    t1 = time()
    m = NBeats(**model_parameters)
    metrics_tr = m.fit(tr, freq=freq)
    t2 = time()
    training_time = t2 - t1
    future_vl = m.make_future_dataframe(vl, n_historic_predictions=test_size)
    t1 = time()
    forecast_vl = m.predict(future_vl)
    t2 = time()
    predicting_time = t2 - t1
    # conf = m.config_train.__dict__
    # lr = conf["learning_rate"]
    # n_epochs = conf["epochs"]
    y_rolled, y_pred_rolled = _return_prediction_from_fold(
        forecast_vl, vl, model_parameters["n_forecasts"], test_size
    )
    return (
        (y_rolled, y_pred_rolled),
        {"training_time": [training_time], "predicting_time": [predicting_time],},
    )


def _train_predict_deepar(tr, vl, model_parameters, freq, test_size):
    t1 = time()
    m = DeepAR(**model_parameters)
    metrics_tr = m.fit(tr, freq=freq)
    t2 = time()
    training_time = t2 - t1
    future_vl = m.make_future_dataframe(vl, n_historic_predictions=test_size)
    t1 = time()
    forecast_vl = m.predict(future_vl)
    t2 = time()
    predicting_time = t2 - t1
    y_rolled, y_pred_rolled = _return_prediction_from_fold(
        forecast_vl, vl, model_parameters["n_forecasts"], test_size
    )
    return (
        (y_rolled, y_pred_rolled),
        {"training_time": [training_time], "predicting_time": [predicting_time],},
    )


def _train_predict_tft(tr, vl, model_parameters, freq, test_size):
    t1 = time()
    m = TFT(**model_parameters)
    metrics_tr = m.fit(tr, freq=freq)
    t2 = time()
    training_time = t2 - t1
    future_vl = m.make_future_dataframe(vl, n_historic_predictions=test_size)
    t1 = time()
    forecast_vl = m.predict(future_vl)
    t2 = time()
    predicting_time = t2 - t1
    y_rolled, y_pred_rolled = _return_prediction_from_fold(
        forecast_vl, vl, model_parameters["n_forecasts"], test_size
    )
    return (
        (y_rolled, y_pred_rolled),
        {"training_time": [training_time], "predicting_time": [predicting_time],},
    )


#
# def _train_predict_np(tr, vl, model_parameters, freq, test_size):
#     t1 = time()
#     m = NeuralProphet(**model_parameters)
#     metrics_tr = m.fit(tr, freq=freq, progress_bar=False)
#     t2 = time()
#     training_time = t2 - t1
#     future_vl = m.make_future_dataframe(vl, n_historic_predictions=test_size)
#     t1 = time()
#     forecast_vl = m.predict(future_vl)
#     t2 = time()
#     predicting_time = t2 - t1
#     conf = m.config_train.__dict__
#     lr = conf["learning_rate"]
#     n_epochs = conf["epochs"]
#     y_rolled, y_pred_rolled = _return_prediction_from_fold(
#         forecast_vl, vl, model_parameters["n_forecasts"], test_size
#     )
#     return (
#         (y_rolled, y_pred_rolled),
#         {
#             "lr": [lr],
#             "n_epoch": [n_epochs],
#             "training_time": [training_time],
#             "predicting_time": [predicting_time],
#         },
#     )


def _train_predict_sklearn(tr, vl, reg, n_forecasts):
    X_tr, y_tr = tr
    X_vl, y_vl = vl
    t1 = time()
    reg.fit(X_tr, y_tr)
    t2 = time()
    training_time = t2 - t1

    t1 = time()
    y_pred_rf = _prediction_vanilla(reg, X_vl, y_vl, n_forecasts)
    t2 = time()
    predicting_time = t2 - t1

    y_rolled = np.array(
        [y_vl[i : i + n_forecasts] for i in range(len(y_vl) - n_forecasts + 1)]
    )
    return (
        (y_rolled, y_pred_rf),
        {"training_time": [training_time], "predicting_time": [predicting_time],},
    )


def _train_predict_arima(tr, vl, model_parameters, test_size, n_forecasts):
    tr_arima = tr.append(
        vl.iloc[vl[vl.ds == tr.ds.iloc[-1]].index.values[0] + 1 : -test_size]
    ).reset_index(drop=True)
    vl_arima = vl.iloc[-test_size:].reset_index(drop=True)
    tr_arima = tr_arima.drop("ds", axis=1)
    vl_arima = vl_arima.drop("ds", axis=1)
    vl_arima.index = range(len(tr_arima), len(tr_arima) + len(vl_arima))
    tr_arima = tr_arima.iloc[-1000:]
    t1 = time()
    model_parameters.update({"endog": tr_arima})
    model = ARIMA(**model_parameters)
    model_arima = model.fit()
    t2 = time()
    training_time = t2 - t1
    t1 = time()
    model_arima_copy = copy(model_arima)
    y_pred_rolled = []
    y_pred_rolled.append(model_arima_copy.forecast(steps=n_forecasts).values)
    for i in range(len(vl_arima) - n_forecasts):
        model_arima_copy = model_arima_copy.append(vl_arima.iloc[i : i + 1])
        y_pred_rolled.append(
            np.array(model_arima_copy.forecast(steps=n_forecasts).values)
        )
    t2 = time()
    predicting_time = t2 - t1

    y = np.array(vl[-test_size:]["y"])
    y_rolled = [y[i : i + n_forecasts] for i in range(len(y) - n_forecasts + 1)]
    y_pred_arima = np.array(y_pred_rolled)

    return (
        (y_rolled, y_pred_arima),
        {"training_time": [training_time], "predicting_time": [predicting_time],},
    )


def _train_predict_prophet(tr, vl, model_parameters, test_size, freq):
    model_parameters
    t1 = time()
    with suppress_stdout_stderr():
        reg_pr = Prophet(**model_parameters).fit(tr)
    t2 = time()
    training_time = t2 - t1
    t1 = time()
    future = reg_pr.make_future_dataframe(
        periods=test_size, freq=freq, include_history=False
    )
    forecast = reg_pr.predict(future)
    y_pred_prophet = np.array(forecast.yhat)
    t2 = time()
    predicting_time = t2 - t1

    y_true = vl.iloc[-test_size:].y

    return (
        (y_true, y_pred_prophet),
        {"training_time": [training_time], "predicting_time": [predicting_time],},
    )


def _train_predict_sarima(tr, vl, model_parameters, test_size, n_forecasts):
    tr_arima = tr.append(
        vl.iloc[vl[vl.ds == tr.ds.iloc[-1]].index.values[0] + 1 : -test_size]
    ).reset_index(drop=True)
    vl_arima = vl.iloc[-test_size:].reset_index(drop=True)
    tr_arima = tr_arima.drop("ds", axis=1)
    vl_arima = vl_arima.drop("ds", axis=1)
    vl_arima.index = range(len(tr_arima), len(tr_arima) + len(vl_arima))
    tr_arima = tr_arima.iloc[-1000:]
    t1 = time()

    model_parameters.update({"endog": tr_arima})
    model = SARIMAX(**model_parameters)
    model_sarima = model.fit()
    t2 = time()
    training_time = t2 - t1
    t1 = time()
    model_sarima_copy = copy(model_sarima)
    y_pred_rolled = []
    y_pred_rolled.append(model_sarima_copy.forecast(steps=n_forecasts).values)
    for i in range(len(vl_arima) - n_forecasts):
        model_sarima_copy = model_sarima_copy.append(vl_arima.iloc[i : i + 1])
        y_pred_rolled.append(
            np.array(model_sarima_copy.forecast(steps=n_forecasts).values)
        )

    t2 = time()
    predicting_time = t2 - t1
    y = np.array(vl[-test_size:]["y"])
    y_rolled = [y[i : i + n_forecasts] for i in range(len(y) - n_forecasts + 1)]
    y_pred_sarima = np.array(y_pred_rolled)

    return (
        (y_rolled, y_pred_sarima),
        {"training_time": [training_time], "predicting_time": [predicting_time],},
    )


def _prediction_vanilla(reg, X_vl, y_vl, n_forecasts):
    X_test = copy(X_vl[: len(y_vl) - n_forecasts + 1])
    predictions = []
    for i in range(0, n_forecasts):
        prediction = reg.predict(X_test)
        predictions.append(prediction.reshape(-1, 1))
        X_test = X_test[:, 1:]
        X_test = np.hstack([X_test, prediction.reshape(-1, 1)])
    return np.hstack(predictions)


def r2_mse_var(y_true, y_pred, var):
    return 1 - (mean_squared_error(y_true, y_pred) / var)


def r2_mae_abs_diff(y_true, y_pred, abs_diff):
    return 1 - (mean_absolute_error(y_true, y_pred) / abs_diff)


def _split_df(df, n_lags, n_forecasts, valid_p=0.2, inputs_overbleed=True):
    """Splits timeseries df into train and validation sets.
    Prevents overbleed of targets. Overbleed of inputs can be configured.
    Args:
        df (pd.DataFrame): data
        n_lags (int): identical to NeuralProhet
        n_forecasts (int): identical to NeuralProhet
        valid_p (float, int): fraction (0,1) of data to use for holdout validation set,
            or number of validation samples >1
        inputs_overbleed (bool): Whether to allow last training targets to be first validation inputs (never targets)
    Returns:
        df_train (pd.DataFrame):  training data
        df_val (pd.DataFrame): validation data
    """
    n_samples = len(df) - n_lags + 2 - (2 * n_forecasts)
    n_samples = n_samples if inputs_overbleed else n_samples - n_lags
    if 0.0 < valid_p < 1.0:
        n_valid = max(1, int(n_samples * valid_p))
    else:
        assert valid_p >= 1
        assert type(valid_p) == int
        n_valid = valid_p
    n_train = n_samples - n_valid
    assert n_train >= 1

    split_idx_train = n_train + n_lags + n_forecasts - 1
    split_idx_val = split_idx_train - n_lags if inputs_overbleed else split_idx_train
    df_train = df.copy(deep=True).iloc[:split_idx_train].reset_index(drop=True)
    df_val = df.copy(deep=True).iloc[split_idx_val:].reset_index(drop=True)
    return df_train, df_val


def _crossvalidation_split_df(
    df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0
):
    """Splits data in k folds for crossvalidation.

    Args:
     df (pd.DataFrame): data
     n_lags (int): identical to NeuralProhet
     n_forecasts (int): identical to NeuralProhet
     k: number of CV folds
     fold_pct: percentage of overall samples to be in each fold
     fold_overlap_pct: percentage of overlap between the validation folds.
         default: 0.0

    Returns:
     list of k tuples [(df_train, df_val), ...] where:
         df_train (pd.DataFrame):  training data
         df_val (pd.DataFrame): validation data
    """
    if n_lags == 0:
        assert n_forecasts == 1
    total_samples = len(df) - n_lags + 2 - (2 * n_forecasts)

    if 0.0 < fold_pct < 1.0:
        samples_fold = max(1, int(fold_pct * total_samples))
    else:
        assert fold_pct >= 1
        assert type(fold_pct) == int
        samples_fold = fold_pct

    samples_overlap = int(fold_overlap_pct * samples_fold)
    assert samples_overlap < samples_fold
    min_train = (
        total_samples - samples_fold - (k - 1) * (samples_fold - samples_overlap)
    )
    assert (
        min_train >= samples_fold
    ), f"min_train = {min_train}, samples_fold = {samples_fold}"
    folds = []
    df_fold = df.copy(deep=True)
    for i in range(k, 0, -1):
        df_train, df_val = _split_df(
            df_fold, n_lags, n_forecasts, valid_p=samples_fold, inputs_overbleed=True
        )
        folds.append((df_train, df_val))
        split_idx = len(df_fold) - samples_fold + samples_overlap
        df_fold = df_fold.iloc[:split_idx].reset_index(drop=True)
    folds = folds[::-1]
    return folds


def preprocess_data_cv(ts, params, n_lags, n_forecasts):
    k = params["k"]
    test_proportion = params["test_proportion"]
    fold_overlap_pct = params["fold_overlap_pct"]
    len_ts = ts.shape[0]
    test_size = int(test_proportion * len_ts)
    method = params["method"].lower()

    np_like_methods = [
        "np",
        "lstm",
        "nbeats",
        "deepar",
        "tft",
        "prophet",
        "arima",
        "sarima",
    ]
    sklearn_methods = ["rf", "gb", "mlp"]

    if method in np_like_methods:
        train, test = _split_df(
            ts, n_lags=n_lags, n_forecasts=n_forecasts, valid_p=test_size
        )
        cv = _crossvalidation_split_df(
            train,
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            k=k,
            fold_pct=test_size,
            fold_overlap_pct=fold_overlap_pct,
        )
        dataset = []
        train_folds = []
        for i, (tr, vl) in enumerate(cv):
            dataset.append([tr.copy(deep=True), vl.copy(deep=True)])
            tr["fold"] = i
            tr["tr/vl"] = "tr"
            vl["fold"] = i
            vl["tr/vl"] = "vl"
            tr = pd.concat([tr, vl])
            train_folds.append(tr)

    elif method in sklearn_methods:
        train, test = _split_df(ts, n_lags=0, n_forecasts=1, valid_p=test_size)
        cv = _crossvalidation_split_df(
            train,
            n_lags=n_lags,
            n_forecasts=1,
            k=k,
            fold_pct=test_size,
            fold_overlap_pct=fold_overlap_pct,
        )
        dataset = []
        train_folds = []
        for i, (tr, vl) in enumerate(cv):
            t = tr.copy()
            v = vl.copy()

            tr["fold"] = i
            tr["tr/vl"] = "tr"
            vl["fold"] = i
            vl["tr/vl"] = "vl"
            tr = pd.concat([tr, vl])
            train_folds.append(tr)

            t["t"] = t["ds"]
            t["y_scaled"] = t["y"]

            v["t"] = v["ds"]
            v["y_scaled"] = v["y"]

            X_tr = tabularize_univariate_datetime(t, n_lags=n_lags, n_forecasts=1)[0][
                "lags"
            ]
            y_tr = tabularize_univariate_datetime(t, n_lags=n_lags, n_forecasts=1)[1]
            X_vl = tabularize_univariate_datetime(v, n_lags=n_lags, n_forecasts=1)[0][
                "lags"
            ]
            y_vl = tabularize_univariate_datetime(v, n_lags=n_lags, n_forecasts=1)[1]

            X_tr = np.vstack([X_tr, X_vl[:-test_size]])
            X_vl = X_vl[-test_size:]
            y_tr = np.vstack([y_tr, y_vl[:-test_size]]).ravel()
            y_vl = y_vl[-test_size:].ravel()

            dataset.append([(X_tr, y_tr), (X_vl, y_vl)])
    return dataset, test_size, pd.concat(train_folds)


def preprocess_data_test(ts, params, n_lags, n_forecasts):
    k = params["k"]
    test_proportion = params["test_proportion"]
    fold_overlap_pct = params["fold_overlap_pct"]
    len_ts = ts.shape[0]
    test_size = int(test_proportion * len_ts)
    method = params["method"].lower()

    np_like_methods = [
        "np",
        "lstm",
        "nbeats",
        "deepar",
        "tft",
        "prophet",
        "arima",
        "sarima",
    ]
    sklearn_methods = ["rf", "gb", "mlp"]

    if method in np_like_methods:
        train, test = _split_df(
            ts, n_lags=n_lags, n_forecasts=n_forecasts, valid_p=test_size
        )

        dataset = []
        train_folds = []
        for i in range(k):
            tr = train.copy(deep=True)
            vl = test.copy(deep=True)
            dataset.append([tr.copy(deep=True), vl.copy(deep=True)])

            tr["fold"] = i
            tr["tr/vl"] = "tr"
            vl["fold"] = i
            vl["tr/vl"] = "vl"
            tr = pd.concat([tr, vl])
            train_folds.append(tr)

    elif method in sklearn_methods:
        train, test = _split_df(ts, n_lags=n_lags, n_forecasts=1, valid_p=test_size)

        dataset = []
        train_folds = []
        for i in range(k):
            tr = train.copy(deep=True)
            vl = test.copy(deep=True)

            t = tr.copy()
            v = vl.copy()

            tr["fold"] = i
            tr["tr/vl"] = "tr"
            vl["fold"] = i
            vl["tr/vl"] = "vl"
            tr = pd.concat([tr, vl])
            train_folds.append(tr)

            t["t"] = t["ds"]
            t["y_scaled"] = t["y"]

            v["t"] = v["ds"]
            v["y_scaled"] = v["y"]

            X_tr = tabularize_univariate_datetime(t, n_lags=n_lags, n_forecasts=1)[0][
                "lags"
            ]
            y_tr = tabularize_univariate_datetime(t, n_lags=n_lags, n_forecasts=1)[1]
            X_vl = tabularize_univariate_datetime(v, n_lags=n_lags, n_forecasts=1)[0][
                "lags"
            ]
            y_vl = tabularize_univariate_datetime(v, n_lags=n_lags, n_forecasts=1)[1]

            X_tr = np.vstack([X_tr, X_vl[:-test_size]])
            X_vl = X_vl[-test_size:]
            y_tr = np.vstack([y_tr, y_vl[:-test_size]]).ravel()
            y_vl = y_vl[-test_size:].ravel()

            dataset.append([(X_tr, y_tr), (X_vl, y_vl)])
    return dataset, test_size, pd.concat(train_folds)
