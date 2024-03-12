"""
Most offline evaluation methods in `neural_networks.model_diagnostics.ModelDiagnostic`
are hard to test because they expect complete set of inputs.
For debugging purposes, use the evaluation functions in `castle_offline_evaluation`.
"""
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from test.testing_utils import set_memory_growth_gpu

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
print(PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test", "output", "test_offline_evaluation")
print(OUTPUT_DIR)

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

try:
    set_memory_growth_gpu()
except RuntimeError:
    logging.warning("GPU growth could not be enabled. "
                    "When running multiple tests, this may be because the physical drivers are already "
                    "initialized, in which case memory growth may already be enabled. "
                    "If memory growth is not enabled, the tests may fail with CUDA error.")


@pytest.mark.parametrize("i_time", [1, "mean"])
@pytest.mark.parametrize("diff", [True, False])
@pytest.mark.parametrize("n_time", [5, False])
@pytest.mark.parametrize("model_diagnostic", ["model_description_castle_adapted", "model_description_castle_original",
                                              "model_description_pre_mask_net",
                                              "model_description_gumbel_softmax_single_output",
                                              "model_description_mask_net"])
def test_plot_plot_double_xy(i_time, diff, n_time, model_diagnostic, request):
    md = request.getfixturevalue(model_diagnostic)

    for var in md.models.keys():
        md.plot_double_xy(i_time, var, diff=diff, nTime=n_time, cmap='RdBu_r',
                          save=OUTPUT_DIR)
        plt.close()
