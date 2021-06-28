from pathlib import PurePosixPath
from typing import Any, Dict, List
from kedro.io.core import (
    AbstractDataSet,
    get_filepath_str,
    get_protocol_and_path,
)

import fsspec
import numpy as np

import pickle
from sktime.utils.data_io import load_from_tsfile_to_dataframe

def _pickle_save(file, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def _pickle_load(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)


class TimeSeriesDataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        """Creates a new instance of TimeSeriesDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the .ts file to load / save data.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> np.ndarray:
        """Loads data from the .ts file.

        Returns:
            Data from the .ts file as a pandas dataframe
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._filepath, self._protocol)
        
        ts_file = load_from_tsfile_to_dataframe(load_path)
        return ts_file
    
    def _save(self, data: np.ndarray) -> None:
        """Saves .ts data to the specified filepath.
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        return None
        
    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset.
        """
        return dict(
            filepath=self._filepath,
            protocol=self._protocol
        )
    
    
    
    
    
    