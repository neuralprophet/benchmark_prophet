import pandas as pd


def _print_dataset_info(
    dataset, dataset_ts_format, freq, test_size, len_ts, dims, n_lags, n_forecasts
):
    print(
        f"""Dataset {dataset} contains {dataset_ts_format.shape[0]} time series
    It has {freq} frequency
    Test size = {test_size}
    Length of time series = {len_ts}
    Number of covariates = {len(dims)}
    Number of lags = {n_lags}
    Number of forecasts = {n_forecasts}"""
    )


def _process(
    dataset_ts_format, params, dataset, test_proportion, preprocessed_datasets
):
    freq = params[dataset]["freq"]
    nan_interpolation = params[dataset]["nan_interpolation"]
    n_ts = dataset_ts_format.shape[0]

    len_ts = pd.DataFrame(dataset_ts_format.iloc[0]["dim_0"]).reset_index().shape[0]
    dims = list(dataset_ts_format.columns)
    test_size = int(test_proportion * len_ts)
    n_lags = params[dataset]["n_lags"]
    n_forecasts = params[dataset]["n_forecasts"]

    for ts in range(n_ts):
        if nan_interpolation == "zero":
            one_time_series = (
                pd.DataFrame(dataset_ts_format.iloc[ts]["dim_0"])
                .reset_index()
                .fillna(0)
            )
        elif nan_interpolation == "linear":
            one_time_series = (
                pd.DataFrame(dataset_ts_format.iloc[ts]["dim_0"])
                .reset_index()
                .interpolate(method="linear")
                .dropna()
            )
        one_time_series.columns = ["ds", "y"]

        preprocessed_datasets.update({f"{dataset}_{ts}": one_time_series})

    _print_dataset_info(
        dataset,
        dataset_ts_format,
        freq,
        test_size,
        len_ts,
        dims,
        n_lags,
        n_forecasts,
    )
    return preprocessed_datasets


def preprocess_into_dataframe(time_series_input, params):
    """Preprocess the data for UCI datasets in NP format.
    Args:
        UCI datasets: Source data. Can be one file if input parameter specified with run.
    Returns:
        Preprocessed data.
    """
    test_proportion = params["test_proportion"]
    if "input" in params.keys():
        uci = False
    else:
        uci = True

    if uci:
        preprocessed_datasets = {}
        for dataset, loader in time_series_input.items():
            dataset_ts_format = loader()
            preprocessed_datasets = _process(
                dataset_ts_format,
                params,
                dataset,
                test_proportion,
                preprocessed_datasets,
            )
        return preprocessed_datasets
    else:
        preprocessed_datasets = {}
        dataset = params["input"]
        dataset_ts_format = time_series_input[dataset]()
        print(dataset_ts_format)
        preprocessed_datasets = _process(
            dataset_ts_format,
            params,
            dataset,
            test_proportion,
            preprocessed_datasets,
        )
        return preprocessed_datasets
