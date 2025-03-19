import os

from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import dili_predict as dp

os.environ["CUDA_VISIBLE_DEVICES"] = "9,10,11,12"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".15"

def main(endpoint):
    folder = dp.path.Experiments.fusion_comparison
    random_state = np.random.RandomState(2623)

    with open(folder / f"{endpoint}_output.log", "w") as f:
        with redirect_stdout(f), redirect_stderr(f):
            print(f"\n # Training model for {endpoint}\n...")

            for include_cp, include_l1000 in [(True, False), (False, True), (True, True)]:
                dp.experiment_runner.early_and_late_fusion(
                    nuts_sampler="blackjax",
                    random_state=random_state,
                    assay_name=endpoint,
                    include_cddd=True,
                    include_cp=include_cp,
                    include_l1000=include_l1000,
                )

            for modality_combination in ("CDDD_CP_L1000", "CDDD_CP", "CDDD_L1000"):
                for strategy in ("butina", "random"):
                    folder = (
                        dp.path.Experiments.fusion_comparison
                        / endpoint
                        / modality_combination
                        / strategy
                    )
                    dp.metrics.consolidate_split_diagnostics(folder, include_plots=True)

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(main, ("CTX", "ROS", "PLD", "MTX_MP", "BSEPi"))
