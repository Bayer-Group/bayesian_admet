# %%
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import matplotlib.pyplot as plt
import umap

import dili_predict as dp

RANDOM_STATE = np.random.RandomState(6556)
IMAGE_FOLDER = Path("/home/gnexx/repos/mlr-bayesian-dili/notebooks/predictive_checks/plots")


def plot_p(idata, group="prior", var_name="p", xlabel="Probability", fontsize=12):
    p_object = idata[group][var_name].values[0] # Samples x observations
    for i in range(p_object.shape[1]):
        sns.kdeplot(p_object[:, i], color="#555555", alpha=0.1)
        sns.despine(left=True)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel("")
    return plt.gcf()


ASSAY_NAME = "CTX"
# %%
data = dp.data.get_modalities(assay_name=ASSAY_NAME)

# %%
cddd = data["CDDD"]
cp = data["CP"]
l1000 = data["L1000"]
all_modalities = pd.concat(
    [
        l1000[dp.data.l1000_feature_columns(l1000)],
        cp[dp.data.cellprofiler_feature_columns(cp)],
        cddd[dp.data.cddd_feature_columns(cddd)],
    ],
    axis=1,
)
# %%
model_cddd = dp.model.horseshoe_logistic(
    X_data=cddd[dp.data.cddd_feature_columns(cddd)],
    y_data=cddd[ASSAY_NAME],
    coefficient_frac=0.1,
)

model_cp = dp.model.horseshoe_logistic(
    X_data=cp[dp.data.cellprofiler_feature_columns(cp)],
    y_data=cp[ASSAY_NAME],
    coefficient_frac=0.1,
)

model_l1000 = dp.model.horseshoe_logistic(
    X_data=l1000[dp.data.l1000_feature_columns(l1000)],
    y_data=l1000[ASSAY_NAME],
    coefficient_frac=0.1,
)

model_all = dp.model.horseshoe_logistic(
    X_data=all_modalities.values, y_data=cddd[ASSAY_NAME],
    coefficient_frac=0.1
)

# %%
prior_cddd = pm.sample_prior_predictive(
    model=model_cddd, samples=1000, random_seed=RANDOM_STATE
)
prior_cp = pm.sample_prior_predictive(
    model=model_cp, samples=1000, random_seed=RANDOM_STATE
)
prior_l1000 = pm.sample_prior_predictive(
    model=model_l1000, samples=1000, random_seed=RANDOM_STATE
)
prior_all = pm.sample_prior_predictive(
    model=model_all, samples=1000, random_seed=RANDOM_STATE
)
# %%
plot_p(prior_cddd)
# %%
plot_p(prior_l1000)
# %%
plot_p(prior_cp)

#%%
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
frame.axes.xaxis.set_ticks([0, 0.5, 1])
plt.axvline(0.5, color="#4e65a6", linestyle=":", alpha=0.7)
plt.savefig(IMAGE_FOLDER / "cp_prior_p.pdf")

plot_p(prior_all)

# %%
reducer = umap.UMAP()
h_l1000 = reducer.fit_transform(l1000[dp.data.l1000_feature_columns(l1000)])
sns.scatterplot(x=h_l1000[:, 0], y=h_l1000[:, 1], hue=l1000[ASSAY_NAME].values)

# %%
h_cp = reducer.fit_transform(cp[dp.data.cellprofiler_feature_columns(cp)])
sns.scatterplot(x=h_cp[:, 0], y=h_cp[:, 1], hue=cp[ASSAY_NAME].values)

# %%
h_cddd = reducer.fit_transform(cddd[dp.data.cddd_feature_columns(cddd)])
sns.scatterplot(x=h_cddd[:, 0], y=h_cddd[:, 1], hue=cddd[ASSAY_NAME].values)

# %%
h_all = reducer.fit_transform(all_modalities)
sns.scatterplot(x=h_all[:, 0], y=h_all[:, 1], hue=cddd[ASSAY_NAME].values)


# %%
with model_cddd:
    trace_cddd = pm.sample(
        1000,
        tune=1000,
        chains=4,
        cores=4,
        random_seed=RANDOM_STATE,
        return_inferencedata=True,
        target_accept=0.95,
    )

# %%
plot_p(trace_cddd, group="posterior")

# %%
with model_cp:
    trace_cp = pm.sample(
        1000,
        tune=1000,
        chains=4,
        cores=4,
        random_seed=RANDOM_STATE,
        return_inferencedata=True,
        target_accept=0.95,
    )
# %%
plot_p(trace_cp, group="posterior")
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
frame.axes.xaxis.set_ticks([0, 0.5, 1])
plt.axvline(0.5, color="#4e65a6", linestyle=":", alpha=0.7)
plt.savefig(IMAGE_FOLDER / "cp_posterior_p.pdf")

# %%
with model_l1000:
    trace_l1000 = pm.sample(
        1000,
        tune=1000,
        chains=4,
        cores=4,
        random_seed=RANDOM_STATE,
        return_inferencedata=True,
        target_accept=0.95,
    )
# %%
plot_p(trace_l1000, group="posterior")

#%%
with model_all:
    trace_all = pm.sample(
        1000,
        tune=1000,
        chains=4,
        cores=4,
        random_seed=RANDOM_STATE,
        return_inferencedata=True,
        target_accept=0.95,
    )


# %%
plot_p(trace_all, group="posterior")
# %%
