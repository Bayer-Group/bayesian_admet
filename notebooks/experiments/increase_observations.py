import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import dili_predict as dp


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".15"


def main(endpoint):
    print(f"Running increase observations experiment for {endpoint}...")
    random_state = np.random.RandomState(76707)
    dp.experiment_runner.increase_observations(
                assay_name=endpoint,
                random_state=random_state,
                nuts_sampler="blackjax",
            )

    for strategy in ("butina", "random"):
        folder = dp.path.Experiments.increase_observations / endpoint / strategy
        dp.metrics.consolidate_split_diagnostics(folder, include_plots=True)

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(main, ("MTX_MP", "PLD", "BSEPi"))
