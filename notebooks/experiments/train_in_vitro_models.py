import numpy as np
from concurrent.futures import ProcessPoolExecutor

import dili_predict as dp

def main(endpoint):
    random_state = np.random.RandomState(203984)
    dp.experiment_runner.train_in_vitro_models(endpoint, random_state=random_state)

if __name__=="__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(main, ("MTX_MP", "PLD", "BSEPi", "ROS", "CTX"))