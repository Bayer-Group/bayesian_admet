# %%
import arviz as az
import pymc as pm
import dili_predict as dp
#%%
if __name__=="__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

    def train_model(assay_name):
        data = dp.data.get_modalities(assay_name=assay_name, include_combinations=True)["ALL"]
        model = dp.model.late_fusion_horseshoe(data, assay_name=assay_name, modalities=["CDDD", "L1000", "CP"])
        with model:
            posterior = pm.sample(nuts_sampler="blackjax", target_accept=0.99, n_init=2000)

        posterior.to_netcdf(dp.path.REPO_ROOT / "notebooks" / "predictive_checks" / f"late_fusion_{assay_name}.nc")

    endpoint = "PLD"
    train_model(endpoint)
    print("Finished training model")

#%%
try:
    #%%
    posterior = az.from_netcdf(dp.path.REPO_ROOT / "notebooks" / "predictive_checks" / f"late_fusion_{endpoint}.nc")

    # %%
    summary = az.summary(posterior).sort_values("r_hat", ascending=False)

    # %%
    # Get names of variables with high r_hat
    (
        summary[summary.r_hat > 1.03]
        .index.str.split("[")
        .to_series()
        .apply(lambda x: x[0])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    #%%
    intercepts = ["alpha_CP", "alpha_L1000", "alpha_CDDD", "alpha"]
    # %%
    az.plot_trace(posterior, var_names="beta", divergences=True)

    # %%
    az.plot_autocorr(posterior, var_names="beta")

    # %%
    az.plot_ess(posterior, var_names="beta", kind="evolution")

    # %%
    az.plot_mcse(posterior, var_names="beta")
    # %%
    az.plot_pair(posterior, var_names=["beta"], divergences=True)

    # %%
    az.plot_parallel(posterior, var_names="beta", figsize=(12, 6))
    # %%
    az.plot_rank(posterior, var_names=["beta", "sigma"], figsize=(12, 6))
    # %%

except FileNotFoundError:
    pass
