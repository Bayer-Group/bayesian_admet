import numpy as np
import pandas as pd

from . import path


def deduplicate_subset(data: pd.DataFrame, assay_name: str) -> pd.DataFrame:
    """
    Returns the deduplicated subset of a pandas DataFrame based on a specified assay name.

    Args:
    data (pd.DataFrame): The input pandas DataFrame containing assay data. It is assumed to contain
        values for assay names as columns and all features. Likely, some assay values are missing
    assay_name (str): The name of the assay used to generate a subset of the data.

    Returns:
    pd.DataFrame: A deduplicated subset of the input data based on the specified assay name.
    """
    return (
        data.dropna(subset=assay_name)
        .groupby("canonical_smiles")
        .mean(numeric_only=True)
    )


def cellprofiler_feature_columns(data: pd.DataFrame) -> list:
    """
    Returns CellProfiler feature columns from a data frame.

    Args:
    data (pd.DataFrame): Input data set contating CellProfiler features.

    Returns:
    list: A list with the feature columns in the data.
    """
    return [
        col
        for col in data.columns
        if col.startswith("Nuclei")
        or col.startswith("Cytoplasm")
        or col.startswith("Cells")
    ]


def l1000_feature_columns(data: pd.DataFrame) -> list:
    """
    Returns L1000 feature columns from a data frame.

    Args:
    data (pd.DataFrame): Input data set contating L1000 features.

    Returns:
    list: A list with the feature columns in the data.
    """
    return [col for col in data.columns if col.endswith("_at")]


def cddd_feature_columns(data: pd.DataFrame) -> list:
    """
    Returns CDDD feature columns from a data frame.

    Args:
    data (pd.DataFrame): Input data set contating CDDD features.

    Returns:
    list: A list with the feature columns in the data.
    """
    return [col for col in data.columns if col.startswith("cddd_")]


feature_extractor = {
    "CDDD": cddd_feature_columns,
    "L1000": l1000_feature_columns,
    "CP": cellprofiler_feature_columns,
}


def clean_cellprofiler_features(
    data: pd.DataFrame,
    assay_name: str,
    nan_count_max: int = 10,
    scaling_factor=20,
    clip_value=1,
):
    feature_columns = cellprofiler_feature_columns(data)

    nan_count = data[feature_columns].isna().sum()
    inf_count = data[feature_columns].abs().eq(np.inf).sum()

    # Remove all columns with more than nan_count_max missing values and inf values
    is_acceptable_nan_count = nan_count <= nan_count_max
    contains_no_infs = inf_count.eq(0)
    valid_feature_columns = nan_count.index[is_acceptable_nan_count & contains_no_infs]
    valid_features = data[valid_feature_columns] / scaling_factor
    # CP tends to have a few very extreme values, which impact model fitting
    valid_features = valid_features.clip(lower=-clip_value, upper=clip_value)
    data = pd.concat([data[assay_name], valid_features], axis=1)

    # There might still be observations with NaNs left.
    data = data.dropna(axis=0)

    if not len(data):
        raise ValueError(
            f"The CellProfiler features data frame is empty after preprocessing. "
            f"Please check the data or adjust the `nan_count_max` argument."
            f" Current value is {nan_count_max=}"
        )

    return data


def preprocess_l1000_features(data: pd.DataFrame, assay_name: str):
    feature_columns = l1000_feature_columns(data)
    preprocessed_features = data[feature_columns] / 1000
    metadata = data[assay_name]
    return pd.concat([metadata, preprocessed_features], axis=1)


def preprocess_cddd_features(data: pd.DataFrame, assay_name: str):
    feature_columns = cddd_feature_columns(data)
    metadata = data[assay_name]
    return pd.concat([metadata, data[feature_columns]], axis=1)


def get_modalities(assay_name, include_combinations=False):
    print(f"Loading data for assay {assay_name}.\n")

    cellprofiler = pd.read_csv(path.CellPainting.publication)
    cellprofiler = deduplicate_subset(cellprofiler, assay_name)
    cellprofiler = clean_cellprofiler_features(cellprofiler, assay_name)
    cellprofiler = cellprofiler[~cellprofiler.index.str.contains("c2[se]n1")]
    cellprofiler = cellprofiler.sort_index()

    l1000 = pd.read_csv(path.L1000.publication)
    l1000 = deduplicate_subset(l1000, assay_name)
    l1000 = preprocess_l1000_features(l1000, assay_name)
    l1000 = l1000[~l1000.index.str.contains("c2[se]n1")] # This smile is currently missing!
    l1000 = l1000[l1000.index.isin(cellprofiler.index)].sort_index()

    cddd = pd.read_csv(path.CDDD.publication)
    cddd = deduplicate_subset(cddd, assay_name)
    cddd = cddd[
        ~cddd.index.str.contains("c2[se]n1")
    ]  # This smile is currently missing!
    cddd = cddd[cddd.index.isin(cellprofiler.index)].sort_index()

    for modality in (cddd, l1000, cellprofiler):
        modality[assay_name] = modality[assay_name].astype(int)

    check_smiles_order(cellprofiler, l1000, cddd)

    cddd_l1000 = pd.concat([cddd, l1000.drop(columns=assay_name)], axis=1)
    cddd_cp = pd.concat([cddd, cellprofiler.drop(columns=assay_name)], axis=1)
    l1000_cp = pd.concat([l1000, cellprofiler.drop(columns=assay_name)], axis=1)

    all_modalities = pd.concat(
        [cddd, l1000.drop(columns=assay_name), cellprofiler.drop(columns=assay_name)],
        axis=1,
    )

    modalities = {
        "CP": cellprofiler,
        "L1000": l1000,
        "CDDD": cddd,
        "CDDD_L1000": cddd_l1000,
        "CDDD_CP": cddd_cp,
        "L1000_CP": l1000_cp,
        "ALL": all_modalities,
    }

    if not include_combinations:
        modalities = {key: modalities[key] for key in ["CP", "L1000", "CDDD"]}

    return modalities


def check_smiles_order(cellprofiler, l1000, cddd):
    cp_l1000_order = cellprofiler.index.equals(l1000.index)
    l1000_cddd_order = l1000.index.equals(cddd.index)

    if not (len(cellprofiler) == len(l1000) == len(cddd)):
        cellprofiler_smiles = set(cellprofiler.index)
        l1000_smiles = set(l1000.index)
        cddd_smiles = set(cddd.index)
        raise ValueError(
            f"The number of observations in the datasets is not the same.\n"
            f"CP: {len(cellprofiler)}, L1000: {len(l1000)}, CDDD: {len(cddd)}."
            f"CP - L1000: {cellprofiler_smiles - l1000_smiles}.\n"
            f"L1000 - CDDD: {l1000_smiles - cddd_smiles}.\n"
        )

    if not cp_l1000_order:
        raise ValueError(
            "The order of the CellProfiler and L1000 data is not the same."
        )

    if not l1000_cddd_order:
        raise ValueError("The order of the L1000 and CDDD data is not the same.")
