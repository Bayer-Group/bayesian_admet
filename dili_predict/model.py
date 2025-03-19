import numpy as np
import pandas as pd

import pymc as pm
import pymc_bart as pmb
from . import data


def binary_dili(
    MTX_MP,
    PLD,
    ROS,
    CTX,
    BSEPi,
    missing_MTX_MP,
    missing_PLD,
    missing_ROS,
    missing_CTX,
    missing_BSEPi,
    p_MTX_MP,
    p_PLD,
    p_ROS,
    p_CTX,
    p_BSEPi,
    DILI_majority,
    scale_intercept=0.5,
    scale_coefficient=0.3,
):
    model = pm.Model()
    with model:
        MTX_MP = pm.Data("MTX_MP", MTX_MP)
        PLD = pm.Data("PLD", PLD)
        ROS = pm.Data("ROS", ROS)
        CTX = pm.Data("CTX", CTX)
        BSEPi = pm.Data("BSEPi", BSEPi)

        DILI = pm.Data("DILI", DILI_majority)

        missing_MTX_MP = pm.Data("missing_MTX_MP", missing_MTX_MP)
        missing_PLD = pm.Data("missing_PLD", missing_PLD)
        missing_ROS = pm.Data("missing_ROS", missing_ROS)
        missing_CTX = pm.Data("missing_CTX", missing_CTX)
        missing_BSEPi = pm.Data("missing_BSEPi", missing_BSEPi)

        p_MTX_MP = pm.Data("p_MTX_MP", p_MTX_MP)
        p_PLD = pm.Data("p_PLD", p_PLD)
        p_ROS = pm.Data("p_ROS", p_ROS)
        p_CTX = pm.Data("p_CTX", p_CTX)
        p_BSEPi = pm.Data("p_BSEPi", p_BSEPi)

        activity_MTX_MP = pm.Bernoulli("activity_MTX_MP", p=p_MTX_MP)
        activity_PLD = pm.Bernoulli("activity_PLD", p=p_PLD)
        activity_ROS = pm.Bernoulli("activity_ROS", p=p_ROS)
        activity_CTX = pm.Bernoulli("activity_CTX", p=p_CTX)
        activity_BSEPi = pm.Bernoulli("activity_BSEPi", p=p_BSEPi)

        alpha = pm.Normal("alpha", mu=0, sigma=scale_intercept)
        beta_MTX_MP = pm.Normal("beta_MTX_MP", mu=0, sigma=scale_coefficient, shape=2)
        beta_PLD = pm.Normal("beta_PLD", mu=0, sigma=scale_coefficient, shape=2)
        beta_ROS = pm.Normal("beta_ROS", mu=0, sigma=scale_coefficient, shape=2)
        beta_CTX = pm.Normal("beta_CTX", mu=0, sigma=scale_coefficient, shape=2)
        beta_BSEPi = pm.Normal("beta_BSEPi", mu=0, sigma=scale_coefficient, shape=2)

        eta_MTX_MP = (1 - missing_MTX_MP) * beta_MTX_MP[
            MTX_MP
        ] + missing_MTX_MP * beta_MTX_MP[activity_MTX_MP]
        eta_PLD = (1 - missing_PLD) * beta_PLD[PLD] + missing_PLD * beta_PLD[
            activity_PLD
        ]
        eta_ROS = (1 - missing_ROS) * beta_ROS[ROS] + missing_ROS * beta_ROS[
            activity_ROS
        ]
        eta_CTX = (1 - missing_CTX) * beta_CTX[CTX] + missing_CTX * beta_CTX[
            activity_CTX
        ]
        eta_BSEPi = (1 - missing_BSEPi) * beta_BSEPi[
            BSEPi
        ] + missing_BSEPi * beta_BSEPi[activity_BSEPi]

        eta = alpha + eta_MTX_MP + eta_PLD + eta_ROS + eta_CTX + eta_BSEPi
        p = pm.Deterministic("p", pm.invlogit(eta))
        pm.Bernoulli("DILI_pred", p=p, observed=DILI)
    return model


def logistic_regression(X_data, y_data, scale_intercept=1, scale_coefficient=0.5):
    model = pm.Model()
    K = X_data.shape[1]
    with model:
        X = pm.Data("X", X_data)
        y = pm.Data("y", y_data)
        alpha_raw = pm.Normal("alpha")
        alpha = alpha_raw * scale_intercept
        beta_raw = pm.Normal("beta_raw", shape=K)
        beta = beta_raw * scale_coefficient
        eta = alpha + X @ beta
        p = pm.Deterministic("p", pm.invlogit(eta))
        pm.Bernoulli("y_pred", p=p, observed=y)
    return model


def horseshoe_logistic(
    X_data,
    y_data=None,
    coefficient_frac=0.1,
    scale_intercept=1,
    nu_tau=1,
    nu_lambda=1,
    slab_scale=1,
    slab_df=2,
    include_likelihood=True,
    model=None,
    suffix="",
):
    """https://arxiv.org/pdf/1707.01694 Appendix C.2
    and https://discourse.mc-stan.org/t/divergent-transitions-with-the-horseshoe-prior/1651/7"""
    if not model:
        model = pm.Model()

    N, D = X_data.shape
    scale_global = coefficient_frac / (1 - coefficient_frac) / pm.math.sqrt(N)
    with model:
        X = pm.Data("X" + suffix, X_data)

        tau_raw = pm.HalfNormal("tau_raw" + suffix, sigma=1)
        tau_scale = pm.InverseGamma(
            "tau_scale" + suffix, alpha=0.5 * nu_tau, beta=0.5 * nu_tau
        )
        tau = tau_raw * pm.math.sqrt(tau_scale) * scale_global

        c_aux = pm.InverseGamma(
            "c_aux" + suffix, alpha=0.5 * slab_df, beta=0.5 * slab_df
        )
        c = slab_scale * pm.math.sqrt(c_aux)
        lambda_raw = pm.HalfNormal("lambda_raw" + suffix, sigma=1, shape=D)
        lambda_scale = pm.InverseGamma(
            "lambda_scale" + suffix, 0.5 * nu_lambda, 0.5 * nu_lambda, shape=D
        )
        lambda_ = lambda_raw * pm.math.sqrt(lambda_scale)
        lambda_tilde = pm.math.sqrt(
            c**2 * pm.math.sqrt(lambda_) / (c**2 + tau**2 * pm.math.sqrt(lambda_))
        )

        alpha = pm.Normal("alpha" + suffix, mu=0, sigma=scale_intercept)
        z = pm.Normal("z" + suffix, mu=0, sigma=1, shape=D)
        beta = pm.Deterministic("beta" + suffix, z * tau * lambda_tilde)
        eta = pm.Deterministic("eta" + suffix, alpha + X @ beta)

        if include_likelihood:
            y = pm.Data("y", y_data)
            p = pm.Deterministic("p", pm.invlogit(eta))
            pm.Bernoulli("y_pred", p=p, observed=y)
    return model


def bart_bernoulli(X_data, y_data, n_trees=50):
    model = pm.Model()
    with model:
        X = pm.Data("X", X_data)
        y = pm.Data("y", y_data)
        p_ = pmb.BART("p_", X=X, Y=y_data, m=n_trees)
        p = pm.Deterministic("p", pm.invlogit(p_))
        pm.Bernoulli("y_pred", p=p, observed=y)
    return model


class GPModel(pm.Model):
    def __init__(self, X_data, y_data, kernel_name, name=""):
        super().__init__(name)
        K = X_data.shape[1]
        X = pm.Data("X", X_data)
        pm.Data("y", y_data)
        cov = self._initialize_kernel(kernel_name, input_dim=K)

        self.gp = pm.gp.Latent(cov_func=cov)
        self.gp.prior(name="f", X=X)

    def _exponential_kernel(self, input_dim):
        self.register_rv(pm.Gamma.dist(alpha=2, beta=1), "ls")
        self.register_rv(pm.Gamma.dist(alpha=2, beta=4), "amplitude")
        return self.amplitude**2 * pm.gp.cov.ExpQuad(input_dim=input_dim, ls=self.ls)

    def _matern52_kernel(self, input_dim):
        self.register_rv(pm.Gamma.dist(alpha=2, beta=1), "ls")
        self.register_rv(pm.Gamma.dist(alpha=2, beta=4), "amplitude")
        return self.amplitude**2 * pm.gp.cov.Matern52(input_dim=input_dim, ls=self.ls)

    def _initialize_kernel(self, kernel_name, input_dim):
        if kernel_name == "exponential":
            return self._exponential_kernel(input_dim)
        elif kernel_name == "matern52":
            return self._matern52_kernel(input_dim)
        raise NotImplementedError(
            f"Supported kernels are 'exponential', 'matern52'. Got {kernel_name}"
        )


def gp_bernoulli(X_data, y_data, kernel_name="matern52"):
    model = GPModel(X_data, y_data, kernel_name=kernel_name)
    with model:
        p = pm.Deterministic(name="p", var=pm.invlogit(model.f))
        pm.Bernoulli("y_pred", p=p, observed=model.y)
    return model


def early_fusion_horseshoe(
    dataset: pd.DataFrame,
    assay_name: str,
    modalities: list[str],
    coefficient_frac: float = 0.03,
):
    fusion_model = pm.Model()
    X = []
    if "CDDD" in modalities:
        X.append(dataset[data.cddd_feature_columns(dataset)])
    if "L1000" in modalities:
        X.append(dataset[data.l1000_feature_columns(dataset)])
    if "CP" in modalities:
        X.append(dataset[data.cellprofiler_feature_columns(dataset)])
    X_data = pd.concat(X, axis=1).values
    y = dataset[assay_name].values

    horseshoe_logistic(
        model=fusion_model,
        X_data=X_data,
        y_data=y,
        coefficient_frac=coefficient_frac,
    )
    return fusion_model


def late_fusion_horseshoe(
    dataset: pd.DataFrame,
    assay_name: str,
    modalities: list[str],
    coefficient_frac=0.03,
    fusion_sigma=0.4,
):
    n_models = len(modalities)
    fusion_model = pm.Model()

    for modality in modalities:
        X_data = dataset[data.feature_extractor[modality](dataset)].values
        horseshoe_logistic(
            model=fusion_model,
            X_data=X_data,
            include_likelihood=False,
            coefficient_frac=coefficient_frac,
            suffix=f"_{modality}",
        )
    y_data = dataset[assay_name].values

    with fusion_model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=fusion_sigma, shape=n_models)
        eta = alpha + sum(
            [
                fusion_model[f"eta_{modality}"] * beta[i]
                for i, modality in enumerate(modalities)
            ]
        )
        y = pm.Data("y", y_data)
        p = pm.Deterministic("p", pm.invlogit(eta))
        pm.Bernoulli("y_pred", p=p, observed=y)

    return fusion_model


def endpoint_models(X_train, y_train, X_val, y_val):
    """Create multiple models for one data set"""
    models = {}
    models["logistic"] = logistic_regression(
        X_data=X_train, y_data=y_train, scale_intercept=1, scale_coefficient=0.01
    )
    models["horseshoe"] = horseshoe_logistic(
        X_data=X_train,
        y_data=y_train,
        scale_intercept=1,
    )
    models["BART"] = bart_bernoulli(X_data=X_train, y_data=y_train)
    models["GP"] = gp_bernoulli(X_data=X_train, y_data=y_train, kernel_name="matern52")

    return {"models": models, "X_val": X_val, "y_val": y_val}


def fit_and_predict(
    model,
    X_val,
    y_val,
    nuts_sampler="pymc",
    target_accept=0.995,
    draws=1000,
    tune=1000,
    random_state: None | np.random.RandomState = None,
):
    """Streamline posterior estimation and prediction"""
    if nuts_sampler not in ("pymc", "blackjax"):
        raise ValueError(
            f"Supported samplers are 'pymc' and 'blackjax'. Passed: {nuts_sampler}"
        )

    def set_data(X_val, y_val):
        if isinstance(X_val, dict):
            for modality, dataset in X_val.items():
                pm.set_data({f"X_{modality}": dataset, "y": y_val})
        else:
            pm.set_data({"X": X_val, "y": y_val})

    with model:
        # GPs have their own way of generating predictions
        if isinstance(model, GPModel):
            if isinstance(X_val, dict):
                raise NotImplementedError("X_val cannot be a dictionary for GP models.")
            posterior = pm.sample(
                nuts_sampler=nuts_sampler,
                target_accept=target_accept,
                random_seed=random_state,
                draws=draws,
                tune=tune,
            )

            pm.compute_log_likelihood(posterior)
            f_pred = model.gp.conditional(name="f_pred", Xnew=X_val)
            p_pred = pm.Deterministic(name="p_pred", var=pm.math.invlogit(f_pred))
            pm.Bernoulli(name="likelihood_pred", p=p_pred)
            posterior_predictive = pm.sample_posterior_predictive(
                posterior, var_names=["f_pred", "p_pred", "likelihood_pred"]
            )
            gp_pred = np.random.binomial(
                n=1, p=posterior_predictive.posterior_predictive["p_pred"]
            )
            posterior_predictive.posterior_predictive["y_pred"] = (
                ("chain", "draw", "obs"),
                gp_pred,
            )
        else:
            if nuts_sampler == "pymc":
                posterior = pm.sample(
                    nuts_sampler=nuts_sampler,
                    nuts_sampler_kwargs={"target_accept": target_accept},
                    random_seed=random_state,
                    draws=draws,
                    tune=tune,
                )
            elif nuts_sampler == "blackjax":
                posterior = pm.sample(
                    nuts_sampler=nuts_sampler,
                    target_accept=target_accept,
                    random_seed=random_state,
                    draws=draws,
                    tune=tune,
                )

            pm.compute_log_likelihood(posterior)
            set_data(X_val, y_val)
            print("Sampling from posterior predictive distribution...")
            posterior_predictive = pm.sample_posterior_predictive(posterior)

        return posterior, posterior_predictive
