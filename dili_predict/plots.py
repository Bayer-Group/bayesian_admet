from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az

import dili_predict as dp

PATH = dp.path.Experiments.model_comparison

MODALITIES = ("CP", "L1000", "CDDD")
SPLITS = ("random", "butina")
ENDPOINTS = ("MTX_MP", "PLD", "BSEPi", "ROS", "CTX")

AXIS_FONTSIZE = 20
LABEL_FONTSIZE = 14
COLORS = {"blue": "#1F77B4", "orange": "#FF7F0E", "gray": "#99999B"}
BW_ADJUST = 1.3


def get_model_comparison_metrics(root_folder, endpoint, modality, split, models):
    root_folder = PATH / endpoint / modality / split / "consolidated"
    lowercase_models = ("horseshoe", "logistic")
    metrics = []
    for model in models:
        model_metrics = pd.read_csv(root_folder / f"{endpoint}_{model}_diagnostics.csv")
        model_metrics["model"] = (
            model.capitalize() if model in lowercase_models else model
        )
        metrics.append(model_metrics)
    return pd.concat(metrics)


def get_modality_comparison_metrics(root_folder, endpoint, split, modalities):
    metrics = []
    for modality in modalities:
        folder = root_folder / endpoint / modality / split / "consolidated"
        modality_metrics = pd.read_csv(folder / f"{endpoint}_horseshoe_diagnostics.csv")
        modality_metrics["modality"] = modality
        metrics.append(modality_metrics)
    return pd.concat(metrics)


def get_increase_comparison_metrics(root_folder, endpoint, split):
    metrics = []
    model_lookup = {"full": "Augmented", "reduced": "Reduced"}
    for model_type in ("full", "reduced"):
        folder = root_folder / endpoint / split / "consolidated"
        classification_metrics = pd.read_csv(
            folder / f"{endpoint}_{model_type}_diagnostics.csv"
        )
        classification_metrics["model_type"] = model_lookup[model_type]
        metrics.append(classification_metrics)
    return pd.concat(metrics)


def get_fusion_comparison_metrics(root_folder, endpoint, split, modality_combination):
    metrics = []
    model_lookup = {"early": "Early", "late": "Late"}
    for fusion in ("early", "late"):
        folder = root_folder / endpoint / modality_combination / split / "consolidated"
        classification_metrics = pd.read_csv(
            folder / f"{endpoint}_{fusion}_diagnostics.csv"
        )
        classification_metrics["fusion"] = model_lookup[fusion]
        metrics.append(classification_metrics)
    single_modality = pd.read_csv(
        dp.path.Experiments.model_comparison
        / endpoint
        / "CDDD"
        / split
        / "consolidated"
        / f"{endpoint}_horseshoe_diagnostics.csv"
    )
    single_modality["fusion"] = "Single"
    metrics.append(single_modality)
    return pd.concat(metrics)


def create_kde_plot(metrics, diagnostic, hue, ax, palette, label_fontsize):
    sns.kdeplot(
        data=metrics,
        x=diagnostic,
        hue=hue,
        ax=ax,
        palette=palette,
        bw_adjust=1.1,
        linewidth=3,
    )
    sns.despine(top=True, right=True, left=True)
    if diagnostic == "MCC":
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    else:
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.get_legend().remove()
    ax.tick_params(axis="both", which="major", labelsize=label_fontsize)


def clean_and_format_labels(
    ax, row_label, col_label, diagnostic, row, col, n_rows, axis_fontsize
):
    if row == 0:
        ax.set_title(col_label, fontsize=axis_fontsize)
    if col == 0:
        ax.set_ylabel(row_label, fontsize=axis_fontsize)
    else:
        ax.set_ylabel("")
    if row == n_rows - 1:
        ax.set_xlabel(diagnostic, fontsize=axis_fontsize)


def create_legend(fig, legend_handles, palette, axis_fontsize):
    custom_handles = [
        Line2D([0], [0], color=palette[handle], lw=2) for handle in legend_handles
    ]
    fig.legend(
        custom_handles,
        legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        fontsize=axis_fontsize,
        bbox_to_anchor=(0.5, 1.005),
    )


def format_and_save_figure(fig, path):
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=2)
    fig.savefig(path, bbox_inches="tight")
    plt.clf()


def model_comparison_figure(diagnostic="MCC"):
    """Figure for Supplement: Model Comparison for Horseshoe, GP and BART"""
    results_path = dp.path.Experiments.model_comparison
    models = ("horseshoe", "BART", "GP")
    palette = {
        "Horseshoe": COLORS["blue"],
        "BART": COLORS["orange"],
        "GP": COLORS["gray"],
    }

    print(f"\nCreating model comparison figure for {diagnostic}...\n")

    for split in SPLITS:
        fig, axes = plt.subplots(
            len(ENDPOINTS), len(MODALITIES), figsize=(15, 20), sharex=True, sharey=True
        )
        for row, endpoint in enumerate(ENDPOINTS):
            print(f"Plotting {endpoint} for {split}")
            for col, modality in enumerate(MODALITIES):
                ax = axes[row, col]
                metrics = get_model_comparison_metrics(
                    results_path, endpoint, modality, split, models
                )
                create_kde_plot(
                    metrics,
                    diagnostic,
                    "model",
                    ax,
                    palette,
                    label_fontsize=LABEL_FONTSIZE,
                )
                clean_and_format_labels(
                    ax=ax,
                    row_label=endpoint,
                    col_label=modality,
                    diagnostic=diagnostic,
                    row=row,
                    col=col,
                    n_rows=len(ENDPOINTS),
                    axis_fontsize=AXIS_FONTSIZE,
                )

        legend_handles = [
            model.capitalize() if model == "horseshoe" else model.upper()
            for model in models
        ]

        create_legend(fig, legend_handles, palette, AXIS_FONTSIZE)
        format_and_save_figure(
            fig,
            dp.path.FIGURES
            / "model_comparison"
            / f"model_comparison_{split}_{diagnostic}.pdf",
        )


def logistic_comparison_figure(diagnostic="MCC"):
    """Figure for Supplement: Model Comparison for Horseshoe, and plain logistic"""

    logistic_models = ("horseshoe", "logistic")
    palette = {
        "Horseshoe": COLORS["blue"],
        "Logistic": COLORS["orange"],
    }
    results_path = dp.path.Experiments.model_comparison

    print(f"\nCreating logistic comparison figure for {diagnostic}...\n")

    for split in SPLITS:
        fig, axes = plt.subplots(
            len(ENDPOINTS), len(MODALITIES), figsize=(15, 20), sharex=True, sharey=True
        )
        for row, endpoint in enumerate(ENDPOINTS):
            print(f"Plotting {endpoint} for {split}")
            for col, modality in enumerate(MODALITIES):
                ax = axes[row, col]
                metrics = get_model_comparison_metrics(
                    results_path, endpoint, modality, split, logistic_models
                )
                create_kde_plot(
                    metrics,
                    diagnostic,
                    "model",
                    ax,
                    palette,
                    label_fontsize=LABEL_FONTSIZE,
                )
                clean_and_format_labels(
                    ax=ax,
                    row_label=endpoint,
                    col_label=modality,
                    diagnostic=diagnostic,
                    row=row,
                    col=col,
                    n_rows=len(ENDPOINTS),
                    axis_fontsize=AXIS_FONTSIZE,
                )

        legend_handles = [model.capitalize() for model in logistic_models]

        create_legend(fig, legend_handles, palette, AXIS_FONTSIZE)
        format_and_save_figure(
            fig,
            dp.path.FIGURES
            / "logistic_comparison"
            / f"logistic_model_comparison_{split}_{diagnostic}.pdf",
        )


def modality_comparison_figure(diagnostic="MCC"):
    """Main Figure: Modality Comparison for CP, L1000, and CDDD"""
    results_path = dp.path.Experiments.model_comparison
    palette_modalities = {
        "CP": COLORS["blue"],
        "L1000": COLORS["orange"],
        "CDDD": COLORS["gray"],
    }

    print(f"\nCreating modality comparison figure for {diagnostic}...\n")

    fig, axes = plt.subplots(
        len(ENDPOINTS), len(SPLITS), figsize=(12, 20), sharex=True, sharey=True
    )

    for row, endpoint in enumerate(ENDPOINTS):
        print(f"Plotting {endpoint}")
        for col, split in enumerate(SPLITS):
            ax = axes[row, col]
            metrics = get_modality_comparison_metrics(
                results_path, endpoint, split, MODALITIES
            )
            create_kde_plot(
                metrics,
                diagnostic,
                "modality",
                ax,
                palette_modalities,
                label_fontsize=LABEL_FONTSIZE,
            )

            split_alias = "cluster" if split == "butina" else "random"
            clean_and_format_labels(
                ax=ax,
                row_label=endpoint,
                col_label=split_alias,
                diagnostic=diagnostic,
                row=row,
                col=col,
                n_rows=len(ENDPOINTS),
                axis_fontsize=AXIS_FONTSIZE,
            )

    create_legend(fig, MODALITIES, palette_modalities, AXIS_FONTSIZE)
    format_and_save_figure(
        fig,
        dp.path.FIGURES
        / "modality_comparison"
        / f"modality_comparison_{diagnostic}.pdf",
    )


def increase_observations_figure(diagnostic="MCC"):
    results_path = dp.path.Experiments.increase_observations
    endpoints = ("MTX_MP", "PLD", "BSEPi")
    palette_increase = {
        "Augmented": COLORS["blue"],
        "Reduced": COLORS["orange"],
    }
    model_types = list(palette_increase.keys())

    print(f"\nCreating increase observation comparison figure for {diagnostic}...\n")

    fig, axes = plt.subplots(
        len(endpoints), len(SPLITS), figsize=(12, 14), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    for row, endpoint in enumerate(endpoints):
        print(f"Plotting {endpoint}")
        for col, split in enumerate(SPLITS):
            ax = axes[row, col]
            metrics = get_increase_comparison_metrics(results_path, endpoint, split)
            create_kde_plot(
                metrics,
                diagnostic,
                "model_type",
                ax,
                palette_increase,
                label_fontsize=LABEL_FONTSIZE,
            )

            split_alias = "cluster" if split == "butina" else "random"
            clean_and_format_labels(
                ax=ax,
                row_label=endpoint,
                col_label=split_alias,
                diagnostic=diagnostic,
                row=row,
                col=col,
                n_rows=len(endpoints),
                axis_fontsize=AXIS_FONTSIZE,
            )

    create_legend(fig, model_types, palette_increase, AXIS_FONTSIZE)
    format_and_save_figure(
        fig,
        dp.path.FIGURES
        / "increase_observations"
        / f"increase_observations_{diagnostic}.pdf",
    )


def fusion_comparison_figure(diagnostic="MCC"):
    modalities_combination = ("CDDD_L1000", "CDDD_CP", "CDDD_CP_L1000")
    palette_increase = {
        "Early": COLORS["blue"],
        "Late": COLORS["orange"],
        "Single": COLORS["gray"],
    }
    model_types = list(palette_increase.keys())

    for combination in modalities_combination:
        print(
            f"\nCreating fusion comparison comparison figure for {diagnostic} and {combination}...\n"
        )
        results_path = dp.path.Experiments.fusion_comparison
        fig, axes = plt.subplots(
            len(ENDPOINTS), len(SPLITS), figsize=(15, 20), sharex=True, sharey=True
        )
        for row, endpoint in enumerate(ENDPOINTS):
            print(f"Plotting {endpoint}")
            for col, split in enumerate(SPLITS):
                ax = axes[row, col]
                metrics = get_fusion_comparison_metrics(
                    results_path, endpoint, split, combination
                )
                create_kde_plot(
                    metrics,
                    diagnostic,
                    "fusion",
                    ax,
                    palette_increase,
                    label_fontsize=LABEL_FONTSIZE,
                )

                split_alias = "cluster" if split == "butina" else "random"
                clean_and_format_labels(
                    ax=ax,
                    row_label=endpoint,
                    col_label=split_alias,
                    diagnostic=diagnostic,
                    row=row,
                    col=col,
                    n_rows=len(ENDPOINTS),
                    axis_fontsize=AXIS_FONTSIZE,
                )

        create_legend(fig, model_types, palette_increase, AXIS_FONTSIZE)
        format_and_save_figure(
            fig,
            dp.path.FIGURES
            / "fusion_comparison"
            / f"fusion_observations_{combination}_{diagnostic}.pdf",
        )


def dili_probability_figure(compound):
    complete_path = dp.path.Experiments.dili_prediction / f"{compound}_full_pp.nc"
    partially_imputed_path = dp.path.Experiments.dili_prediction / f"{compound}_partial_impute_pp.nc"
    imputed_path = dp.path.Experiments.dili_prediction / f"{compound}_impute_pp.nc"

    complete = az.from_netcdf(complete_path)
    partially_imputed = az.from_netcdf(partially_imputed_path)
    imputed = az.from_netcdf(imputed_path)

    experimental = pd.DataFrame({"DILI probability": complete.posterior_predictive["p"].values.flatten()})
    experimental["group"] = "Experimental"
    partially_imputed = pd.DataFrame({"DILI probability": partially_imputed.posterior_predictive["p"].values.flatten()})
    partially_imputed["group"] = "Partially Imputed"
    imputed = pd.DataFrame({"DILI probability": imputed.posterior_predictive["p"].values.flatten()})
    imputed["group"] = "Fully Imputed"

    data = pd.concat([experimental, partially_imputed, imputed])

    g = sns.kdeplot(
        data=data,
        x="DILI probability",
        hue="group",
        bw_adjust=1.1,
        linewidth=2,
        palette = "viridis"
    )
    sns.despine(top=True, right=True, left=True)

    plt.setp(g.get_legend().get_texts(), fontsize=15)
    g.get_legend().set_title(None)

    plt.xlabel('DILI probability', fontsize=15)
    plt.ylabel('Density', fontsize=15)

    plt.tick_params(axis="both", which="major", labelsize=13)

    plt.savefig(dp.path.FIGURES / "dili_predict" / f"{compound}_probability.pdf", bbox_inches="tight")
    plt.clf()


def plot_p(idata, group="prior", var_name="p", xlabel="Probability", fontsize=12):
    p_object = idata[group][var_name].values[0]  # Samples x observations
    for i in range(p_object.shape[1]):
        sns.kdeplot(p_object[:, i], color="#555555", alpha=0.1)
        sns.despine(left=True)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel("")
    return plt.gcf()
