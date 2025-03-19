from typing import Optional
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from dili_predict import data
from dili_predict import path
from dili_predict import splits
from dili_predict.model import (
    early_fusion_horseshoe,
    endpoint_models,
    GPModel,
    fit_and_predict,
    late_fusion_horseshoe,
    horseshoe_logistic,
    binary_dili,
)
from dili_predict.metrics import classification_diagnostics, sampling_diagnostics


def calculate_and_save_diagnostics(
    parent_folder,
    assay_name,
    model_name,
    posterior,
    posterior_predictive,
    y_val,
    split_strategy,
    split_id: str | int = "",
    save_posterior_samples: bool = False,
    save_plots: bool = False,
):
    parent_folder = parent_folder / split_strategy

    if not parent_folder.exists():
        parent_folder.mkdir(parents=True)

    if save_posterior_samples:
        posterior.to_netcdf(
            parent_folder / f"{assay_name}_{model_name}_{split_id}_posterior.nc"
        )
        posterior_predictive.to_netcdf(
            parent_folder
            / f"{assay_name}_{model_name}_{split_id}_posterior_predictive.nc"
        )

    classification_report = classification_diagnostics(y_val, posterior_predictive)
    classification_report.to_csv(
        parent_folder
        / f"{assay_name}_{model_name}_{split_id}_classification_diagnostics.csv",
        index=False,
    )

    sampling_report = sampling_diagnostics(posterior)
    sampling_report.to_csv(
        parent_folder
        / f"{assay_name}_{model_name}_{split_id}_sampling_diagnostics.csv",
        index=False,
    )

    if save_plots:
        classification_report.hist(bins=14, grid=False)
        plt.savefig(
            parent_folder / f"{assay_name}_{model_name}_{split_id}_diagnostics.png"
        )
        plt.clf()


def single_modality(
    assay_name: str,
    reference_modality: str = "CDDD",
    random_state: None | np.random.RandomState = None,
):
    modalities_data = data.get_modalities(assay_name)

    reference_feature_columns = data.feature_extractor[reference_modality](
        modalities_data[reference_modality]
    )

    for strategy in ("butina", "random"):
        print(f"Split strategy with {reference_modality} as a reference: {strategy}\n")

        smile_splits = splits.split(
            X=modalities_data[reference_modality][reference_feature_columns].values,
            y=modalities_data[reference_modality][assay_name],
            strategy=strategy,
            random_state=random_state,
        )

        for split_id, (train_idx, val_idx) in enumerate(smile_splits):
            train_smiles = modalities_data[reference_modality].index[train_idx]
            val_smiles = modalities_data[reference_modality].index[val_idx]
            print(
                f"Train size: {len(train_smiles)}. Validation size: {len(val_smiles)}."
            )

            fit_and_save_single_modality(
                modalities_data=modalities_data,
                train_smiles=train_smiles,
                val_smiles=val_smiles,
                assay_name=assay_name,
                extract_feature_columns=data.feature_extractor,
                random_state=random_state,
                strategy=strategy,
                split_id=split_id,
            )


def fit_and_save_single_modality(
    modalities_data,
    train_smiles,
    val_smiles,
    assay_name,
    extract_feature_columns,
    random_state,
    strategy,
    split_id,
):
    for modality, data_ in modalities_data.items():
        train_data = data_[data_.index.isin(train_smiles)]
        val_data = data_[data_.index.isin(val_smiles)]

        print(f"Creating results for modality {modality}: {assay_name}")
        X_train = train_data[extract_feature_columns[modality](train_data)].values
        y_train = train_data[assay_name].values
        X_val = val_data[extract_feature_columns[modality](val_data)].values
        y_val = val_data[assay_name].values
        model_container = endpoint_models(X_train, y_train, X_val, y_val)

        for model_name, model in model_container["models"].items():
            print(f"\n----- Fitting {model_name}. -----\n")
            nuts_sampler = "blackjax" if isinstance(model, GPModel) else "pymc"
            try:
                posterior, posterior_predictive = fit_and_predict(
                    model,
                    model_container["X_val"],
                    model_container["y_val"],
                    nuts_sampler=nuts_sampler,
                    random_state=random_state,
                    target_accept=0.95,
                )
            except pm.SamplingError:
                print(
                    f"Sampling Error! {model_name} could not be estimated. "
                    "Continue to next model."
                )
                return

            parent = path.Experiments.model_comparison / assay_name / modality

            calculate_and_save_diagnostics(
                parent_folder=parent,
                assay_name=assay_name,
                model_name=model_name,
                posterior=posterior,
                posterior_predictive=posterior_predictive,
                y_val=model_container["y_val"],
                split_strategy=strategy,
                split_id=split_id,
            )


def early_and_late_fusion(
    assay_name: str,
    reference_modality: str = "CDDD",
    nuts_sampler: str = "pymc",
    include_cddd: bool = True,
    include_cp: bool = True,
    include_l1000: bool = True,
    random_state: None | np.random.RandomState = None,
):
    def clean_X_early(X):
        modality_columns = []
        if include_cddd:
            modality_columns += data.cddd_feature_columns(X)
        if include_cp:
            modality_columns += data.cellprofiler_feature_columns(X)
        if include_l1000:
            modality_columns += data.l1000_feature_columns(X)
        return X[modality_columns].values

    def clean_X_late(X, modalities):
        X_lookup = {
            "CDDD": X[data.cddd_feature_columns(X)].values,
            "CP": X[data.cellprofiler_feature_columns(X)].values,
            "L1000": X[data.l1000_feature_columns(X)].values,
        }
        return {k: v for k, v in X_lookup.items() if k in modalities}

    modalities = []
    if include_cddd:
        modalities.append("CDDD")
    if include_cp:
        modalities.append("CP")
    if include_l1000:
        modalities.append("L1000")

    assert reference_modality in modalities

    print(f"Running experiment for modality combination: {'_'.join(modalities)}\n")

    data_ = data.get_modalities(assay_name, include_combinations=True)["ALL"]

    reference_feature_columns = data.feature_extractor[reference_modality](data_)

    for strategy in ("butina", "random"):
        print(f"Split strategy with {reference_modality} as a reference: {strategy}\n")
        smile_splits = splits.split(
            X=data_[reference_feature_columns].values,
            y=data_[assay_name],
            strategy=strategy,
            random_state=random_state,
        )

        for split_id, (train_idx, val_idx) in enumerate(smile_splits):
            train_smiles = data_.index[train_idx]
            val_smiles = data_.index[val_idx]
            print(
                f"Train size: {len(train_smiles)}. Validation size: {len(val_smiles)}."
            )

            train_data = data_[data_.index.isin(train_smiles)]
            val_data = data_[data_.index.isin(val_smiles)]
            y_val = val_data[assay_name].values

            models = {
                "early": early_fusion_horseshoe(
                    train_data, assay_name, modalities=modalities
                ),
                "late": late_fusion_horseshoe(
                    train_data, assay_name, modalities=modalities
                ),
            }

            for model_name, model in models.items():
                print(f"\n----- Fitting {model_name}. -----\n")
                match model_name:
                    case "early":
                        X_val = clean_X_early(val_data)

                    case "late":
                        X_val = clean_X_late(val_data, modalities)
                try:
                    posterior, posterior_predictive = fit_and_predict(
                        model,
                        X_val,
                        y_val,
                        nuts_sampler=nuts_sampler,
                        random_state=random_state,
                        target_accept=0.999,
                    )
                except pm.SamplingError:
                    print(
                        f"Sampling Error! {model_name} could not be estimated. "
                        "Continue to next model."
                    )
                    continue

                parent = (
                    path.Experiments.fusion_comparison
                    / assay_name
                    / "_".join(modalities)
                )
                calculate_and_save_diagnostics(
                    parent_folder=parent,
                    assay_name=assay_name,
                    model_name=model_name,
                    posterior=posterior,
                    posterior_predictive=posterior_predictive,
                    y_val=y_val,
                    split_strategy=strategy,
                    split_id=split_id,
                )


def increase_observations(
    assay_name: str,
    random_state: Optional[np.random.RandomState] = None,
    nuts_sampler: str = "pymc",
):
    reduced_data = data.get_modalities(assay_name=assay_name)["CDDD"]
    full_data = pd.read_csv(path.CDDD.publication)
    full_data = full_data[full_data[assay_name].notna()].set_index("canonical_smiles")
    full_data = data.deduplicate_subset(full_data, assay_name)
    full_data = full_data[~full_data.index.str.contains("c2[se]n1")]

    for strategy in ("butina", "random"):
        print(f"Split strategy with CDDD as a reference: {strategy}\n")
        smile_splits = splits.split(
            X=reduced_data[data.cddd_feature_columns(reduced_data)].values,
            y=reduced_data[assay_name],
            strategy=strategy,
            random_state=random_state,
        )

        for split_id, (train_idx, val_idx) in enumerate(smile_splits):
            train_smiles = reduced_data.index[train_idx]
            val_smiles = reduced_data.index[val_idx]
            print(
                f"Train size: {len(train_smiles)}. Validation size: {len(val_smiles)}."
            )

            reduced_train = reduced_data[reduced_data.index.isin(train_smiles)]
            reduced_validation = reduced_data[reduced_data.index.isin(val_smiles)]

            full_train = full_data[~full_data.index.isin(val_smiles)]
            full_validation = full_data[full_data.index.isin(val_smiles)]

            datasets = {
                "full": (full_train, full_validation),
                "reduced": (reduced_train, reduced_validation),
            }

            for dataset_name, (train_data, val_data) in datasets.items():
                print(f"Creating results for {dataset_name} dataset: {assay_name}")
                X_data = train_data[data.cddd_feature_columns(train_data)].values
                y_data = train_data[assay_name].values
                X_val = val_data[data.cddd_feature_columns(val_data)].values
                y_val = val_data[assay_name].values
                print(
                    f"Training: Train size: {len(X_data)}. Validation size: {len(X_val)}"
                )
                model = horseshoe_logistic(X_data, y_data)
                try:
                    posterior, posterior_predictive = fit_and_predict(
                        model,
                        X_val,
                        y_val,
                        nuts_sampler=nuts_sampler,
                        random_state=random_state,
                        target_accept=0.995,
                    )
                except pm.SamplingError:
                    print(
                        f"Sampling Error! Model for {dataset_name} could not be estimated. "
                        "Continue to next model."
                    )
                    continue

                parent = path.Experiments.increase_observations / assay_name
                calculate_and_save_diagnostics(
                    parent_folder=parent,
                    assay_name=assay_name,
                    model_name=dataset_name,
                    posterior=posterior,
                    posterior_predictive=posterior_predictive,
                    y_val=y_val,
                    split_strategy=strategy,
                    split_id=split_id,
                )


def dili_experiment(random_state: np.random.RandomState | None = None):
    endpoints = pd.read_csv(path.DATA_PUBLICATION / "dili_with_names.csv")
    endpoints = endpoints.drop(columns=["canonical_smiles", "name"]).astype(int)

    model = binary_dili(
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
    )

    with model:
        posterior = pm.sample(random_seed=random_state)

    posterior.to_netcdf(path.Experiments.dili_prediction / "dili_posterior.nc")


def train_in_vitro_models(
    assay_name: str, random_state: np.random.RandomState | None = None
):
    cddd = data.get_modalities(assay_name)["CDDD"]
    X_data = cddd[data.cddd_feature_columns(cddd)].values
    y_data = cddd[assay_name].values
    model = horseshoe_logistic(X_data=X_data, y_data=y_data, nu_lambda=1, nu_tau=1)
    posterior, posterior_predictive = fit_and_predict(
        model, X_data, y_data, random_state=random_state, target_accept=0.99999
    )
    posterior.to_netcdf(path.Experiments.dili_prediction / f"{assay_name}_posterior.nc")
    posterior_predictive.to_netcdf(
        path.Experiments.dili_prediction / f"{assay_name}_posterior_predictive.nc"
    )


def predict_for_smiles(
    assay_name, canonical_smiles, random_state: np.random.RandomState | None = None
):
    posterior = az.from_netcdf(
        path.Experiments.dili_prediction / f"{assay_name}_posterior.nc"
    )
    cddd = pd.read_csv(path.CDDD.publication)
    cddd = cddd[cddd.canonical_smiles == canonical_smiles]

    model = horseshoe_logistic(
        X_data=cddd[data.cddd_feature_columns(cddd)].values,
        y_data=cddd[assay_name].values,
    )

    with model:
        posterior_predictive = pm.sample_posterior_predictive(
            posterior, model, random_seed=random_state
        )

    # Return proportion of ones from Bernoulli samples
    return posterior_predictive.posterior_predictive.y_pred.mean()
