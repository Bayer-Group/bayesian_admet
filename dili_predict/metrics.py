from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az


def classification_diagnostics(
    y_true,
    posterior_predictive: az.InferenceData,
    group: str = "posterior_predictive",
    var_names: str = "y_pred",
) -> pd.DataFrame:
    y_pred = pd.DataFrame(
        az.extract(posterior_predictive, group=group, var_names=var_names).values.T
    )

    diagnostics = pd.DataFrame()
    diagnostics["MCC"] = y_pred.apply(
        lambda row: matthews_corrcoef(y_true, row), axis=1
    )
    diagnostics["Balanced Accuracy"] = y_pred.apply(
        lambda row: balanced_accuracy_score(y_true, row), axis=1
    )
    diagnostics["Precision"] = y_pred.apply(
        lambda row: precision_score(y_true, row, zero_division=0), axis=1
    )
    diagnostics["Recall"] = y_pred.apply(lambda row: recall_score(y_true, row), axis=1)
    diagnostics["F1 Score"] = y_pred.apply(lambda row: f1_score(y_true, row), axis=1)

    return diagnostics


def consolidate_split_diagnostics(folder: Path, include_plots: bool = False):
    consolidated_folder = folder / "consolidated"
    if not consolidated_folder.exists():
        consolidated_folder.mkdir(parents=True)

    sampling_statistics_files = folder.glob("*sampling_diagnostics.csv")
    sampling_diagnostics = []
    for file in sampling_statistics_files:
        # Format is <endpoint>_<model>_<split_id>_sampling_diagnostics.csv
        splits = file.name.split("_")[:-2]
        if len(splits) == 4:
            # Takes into account when endpoint includes underscore, e.g., "MTX_MP"
            splits = [splits[0] + "_" + splits[1], splits[2], splits[3]]
        endpoint, model, split = splits
        diagnostics = pd.read_csv(file)
        diagnostics["endpoint"] = endpoint
        diagnostics["model"] = model
        diagnostics["split"] = split
        sampling_diagnostics.append(diagnostics)
    sampling_diagnostics = pd.concat(sampling_diagnostics).reset_index(drop=True)
    sampling_diagnostics.sort_values(["model", "split", "highest_r_hat"]).to_csv(
        consolidated_folder / "sampling_diagnostics.csv", index=False
    )

    diagnostic_files = []
    classification_files = folder.glob("*classification_diagnostics.csv")
    for file in classification_files:
        # Format is <endpoint>_<model>_<split_id>_classification_diagnostics.csv
        splits = file.name.split("_")[:-2]
        if len(splits) == 4:
            # Takes into account when endpoint includes underscore, e.g., "MTX_MP"
            splits = [splits[0] + "_" + splits[1], splits[2], splits[3]]
        diagnostic_files.append(splits)

    endpoint_name = diagnostic_files[0][0]
    diagnostic_files = pd.DataFrame(
        diagnostic_files, columns=["endpoint", "model", "split"]
    )

    models = diagnostic_files["model"].unique()
    for model in models:
        print(f"Consolidating metrics for {model}")
        model_files = diagnostic_files[diagnostic_files["model"] == model]
        model_diagnostics = []
        for _, file in model_files.iterrows():
            diagnostics = pd.read_csv(
                folder
                / f"{file['endpoint']}_{file['model']}_{file['split']}_classification_diagnostics.csv"
            )
            diagnostics["split"] = file["split"]
            model_diagnostics.append(diagnostics)
        model_diagnostics = pd.concat(model_diagnostics)
        model_diagnostics.to_csv(
            consolidated_folder / f"{endpoint_name}_{model}_diagnostics.csv",
            index=False,
        )

        metrics_quantiles(model_diagnostics).to_csv(
            consolidated_folder / f"{endpoint_name}_{model}_diagnostics_quantiles.csv"
        )

        if include_plots:
            print("Creating and saving plot...")
            model_diagnostics = model_diagnostics.melt(
                id_vars="split", var_name="metric", value_name="value"
            )
            g = sns.FacetGrid(
                model_diagnostics,
                col="metric",
                hue="split",
                col_wrap=2,
                aspect=1.5,
                palette="colorblind",
            )
            g.map(sns.kdeplot, "value", fill=False, bw_adjust=1.5)
            g.despine()
            g.savefig(consolidated_folder / f"{endpoint_name}_{model}_diagnostics.pdf")
            plt.clf()


def sampling_diagnostics(posterior):
    try:
        n_divergences = posterior.sample_stats.diverging.values.sum()
    except AttributeError:
        # BART has no "diverging" attribute, and usually has no divergences
        n_divergences = 0

    highest_r_hat = az.summary(posterior).sort_values("r_hat")["r_hat"].iloc[-1]

    return pd.DataFrame(
        columns=["n_divergences", "highest_r_hat"],
        data=[[n_divergences, highest_r_hat]],
    )


def metrics_quantiles(metrics, lower=0.03, upper=0.97):
    metrics = metrics.drop(columns=["split"], errors="ignore")
    quantiles = [lower, 0.25, 0.5, 0.75, upper]
    summary = metrics.quantile(quantiles)
    summary.index = summary.index.map(lambda x: f"{int(x * 100)}%")
    return summary
