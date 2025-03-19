from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from .cluster import butina_cluster


def split(
    X,
    y,
    strategy,
    n_splits=5,
    random_state=None,
    butina_threshold=0.5,
    butina_distance="cosine",
):
    if strategy not in ("random", "butina"):
        raise NotImplementedError("Only 'random' and 'butina' strategies are supported")

    if strategy == "random":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return kf.split(X=X, y=y)
    if strategy == "butina":
        butina_clusters = butina_cluster(
            X, threshold=butina_threshold, distance=butina_distance
        )
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return sgkf.split(X=X, y=y, groups=butina_clusters)
