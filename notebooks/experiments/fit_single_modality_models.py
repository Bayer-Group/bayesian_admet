import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import dili_predict as dp

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".15"


def main(endpoint):
    print(f"Running single modality model experiment for {endpoint}...")
    random_state = np.random.RandomState(6584)
    dp.experiment_runner.single_modality(endpoint, random_state=random_state)

    for strategy in ("butina", "random"):
        for modality in ("CP", "CDDD", "L1000"):
            folder = dp.path.Experiments.model_comparison / endpoint / modality / strategy
            dp.metrics.consolidate_split_diagnostics(folder, include_plots=True)


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(main, ("CTX", "ROS", "PLD", "MTX_MP", "BSEPi"))
