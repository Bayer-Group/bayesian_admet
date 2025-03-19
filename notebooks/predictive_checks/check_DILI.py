#%%
import numpy as np
import pandas as pd
import pymc as pm

import dili_predict as dp

endpoints = pd.read_csv(dp.path.CDDD.publication)
endpoints = endpoints[["MTX_MP", "PLD", "CTX", "ROS", "BSEPi", "DILI_majority"]]
endpoints = endpoints.dropna().astype(int)

#%%
model = dp.model.binary_dili(
    MTX_MP=endpoints["MTX_MP"],
    PLD=endpoints["PLD"],
    CTX=endpoints["CTX"],
    ROS=endpoints["ROS"],
    BSEPi=endpoints["BSEPi"],
    DILI_majority=endpoints["DILI_majority"],
    missing_MTX_MP=np.zeros(len(endpoints)),
    missing_PLD=np.zeros(len(endpoints)),
    missing_CTX=np.zeros(len(endpoints)),
    missing_ROS=np.zeros(len(endpoints)),
    missing_BSEPi=np.zeros(len(endpoints)),
    p_MTX_MP=np.zeros(len(endpoints)),
    p_PLD=np.zeros(len(endpoints)),
    p_CTX=np.zeros(len(endpoints)),
    p_ROS=np.zeros(len(endpoints)),
    p_BSEPi=np.zeros(len(endpoints)),
    scale_coefficient=0.3,
    scale_intercept=0.5,
)

ppc = pm.sample_prior_predictive(samples=1000, model=model)

dp.plots.plot_p(ppc)
# %%
