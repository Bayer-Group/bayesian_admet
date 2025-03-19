# %%
import matplotlib.pyplot as plt
import dili_predict as dp
import pymc as pm


ASSAY_NAME = "CTX"
MODALITY = "CDDD"
RANDOM_STATE = 365
feature_column_selector = {"CDDD": dp.data.cddd_feature_columns,
                           "L1000": dp.data.l1000_feature_columns,
                           "CP": dp.data.cellprofiler_feature_columns}
FEATURE_EXTRACTOR = feature_column_selector[MODALITY]

#%%
data = dp.data.get_modalities(assay_name=ASSAY_NAME)

# %%
horseshoe = dp.model.horseshoe_logistic(
    X_data=data[MODALITY][FEATURE_EXTRACTOR(data[MODALITY])],
    y_data=data[MODALITY][ASSAY_NAME],
    coefficient_frac=0.1,
    scale_intercept=1,
    nu_lambda=1,
    nu_tau=1,
)

horseshoe_wide = dp.model.horseshoe_logistic(
    X_data=data[MODALITY][FEATURE_EXTRACTOR(data[MODALITY])],
    y_data=data[MODALITY][ASSAY_NAME],
    coefficient_frac=0.1,
    scale_intercept=3,
    nu_lambda=1,
    nu_tau=1,
)

prior = pm.sample_prior_predictive(model=horseshoe, samples=1000, random_seed=RANDOM_STATE)
prior_wide = pm.sample_prior_predictive(
    model=horseshoe_wide, samples=1000, random_seed=RANDOM_STATE
)

#%%
dp.plots.plot_p(prior)
plt.savefig(dp.path.FIGURES / "predictive_checks" / "ppc_narrow.pdf")

#%%
dp.plots.plot_p(prior_wide)
plt.savefig(dp.path.FIGURES / "predictive_checks" / "ppc_wide.pdf")
# %%
