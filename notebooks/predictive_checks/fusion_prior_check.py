# %%
import dili_predict as dp
import pymc as pm
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt

#%%
def plot_p(idata, group="prior", var_name="p", xlabel="Probability", fontsize=12):
    p_prior = idata[group][var_name].values[0]
    for i in range(p_prior.shape[1]):
        sns.kdeplot(p_prior[:, i], color="#555555", alpha=0.1)
        sns.despine(left=True)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel("")
    return plt.gcf()

# %%
assay_name = "PLD"
data = dp.data.get_modalities(assay_name, include_combinations=True)["ALL"]

#%%
early = dp.model.early_fusion_horseshoe(data, assay_name)
late = dp.model.late_fusion_horseshoe(data, assay_name, coefficient_frac=0.03, fusion_sigma=0.4)
#%%
prior_early = pm.sample_prior_predictive(model=early)
prior_late = pm.sample_prior_predictive(model=late)
# %%
plot_p(prior_early)

# %%
plot_p(prior_late)
# %%
