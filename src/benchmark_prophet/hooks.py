# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.versioning import Journal
from benchmark_prophet.pipelines.data_preprocessing import pipeline as dp
from benchmark_prophet.pipelines.cross_validation import pipeline as cv
# from benchmark_prophet.pipelines.testing import pipeline as test
from benchmark_prophet.io.datasets.ts_dataset import TimeSeriesDataSet

import warnings

warnings.filterwarnings("ignore")


class ProjectHooks:
    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any], pipeline, catalog):
        """A hook implementation to add a catalog entry
        based on the filename passed to the command line, e.g.:
            kedro run --params=input:time_series_1.ts
            kedro run --params=input:time_series_2.ts
            kedro run --params=input:time_series_3.ts
        """
        try:
            filename = run_params["extra_params"]["input"]
            # add input dataset
            input_dataset_name = "time_series_input"
            input_dataset = TimeSeriesDataSet(filepath=f"data/01_raw/{filename}.ts")
            catalog.add(input_dataset_name, input_dataset)
        except:
            pass

    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.

        """

        return {"__default__": Pipeline([])}

    @hook_impl
    def register_config_loader(self, conf_paths: Iterable[str]) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )

    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.

        """
        dp_pipeline = dp.create_pipeline()
        cv_pipeline = cv.create_pipeline()
        # test_pipeline = test.create_pipeline()


        return {
            "dp": dp_pipeline,
            "cv": cv_pipeline,
            # "test": test_pipeline,
            "__default__": dp_pipeline
            + cv_pipeline
            # + test_pipeline
        }


project_hooks = ProjectHooks()
