#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP

import dili_predict as dp

endpoints = ("MTX_MP", "PLD", "BSEPi", "ROS", "CTX")
modalities = ("CDDD", "L1000", "CP")
data = {endpoint: dp.data.get_modalities(endpoint) for endpoint in endpoints}

#%%
def get_umap_features(embeddings, endpoint, modality):
    feature_selector = {
        "CDDD": dp.data.cddd_feature_columns,
        "L1000": dp.data.l1000_feature_columns,
        "CP": dp.data.cellprofiler_feature_columns,
    }

    umap = UMAP(n_components=2, metric="cosine", min_dist=0.25, random_state=42)
    feature_columns = feature_selector[modality](embeddings)
    features = umap.fit_transform(embeddings[feature_columns].values)
    features = pd.DataFrame(features, columns=["UMAP1", "UMAP2"])
    features["label"] = embeddings[endpoint].values
    features["modality"] = modality
    features["assay"] = endpoint
    features["canonical_smiles"] = embeddings.index
    return features



umap_df = []

for endpoint, assays in data.items():
    for modality in modalities:
        embeddings = data[endpoint][modality]
        umap_df.append(get_umap_features(embeddings, endpoint, modality))

umap_df = pd.concat(umap_df)

# %%
g = sns.FacetGrid(umap_df, col="assay", row="modality", hue="label", height=4, sharex=False, sharey=False)
g.map(plt.scatter, "UMAP1", "UMAP2", alpha=0.8)

# Increase title size
g.set_titles(col_template="{col_name}", row_template="{row_name}", size=22)

# Increase axis labels size
g.set_axis_labels("UMAP1", "UMAP2", fontsize=20)

# Increase tick size
for ax in g.axes.flatten():
    ax.tick_params(axis='both', which='major', labelsize=12)

g.fig.subplots_adjust(hspace=0.2, wspace=0.2)


# Place legend on top
g.add_legend(title="", bbox_to_anchor=(0.45, 1.05), loc='center', fontsize=18, ncol=2)
legend = g.legend
legend.set_title("")
for new_label, text in zip(("inactive", "active"), legend.get_texts()):
    text.set_text(new_label)
    text.set_fontsize(20)


g.fig.set_size_inches(18, 12)
g.savefig(dp.path.FIGURES / "umap_plot.pdf")# %%# %%

# %%
