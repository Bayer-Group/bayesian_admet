# %%
import numpy as np
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from joblib import Parallel, delayed

N_JOBS = 10


def butina_cluster(embeddings: np.ndarray, threshold: float = 0.4, distance="cosine"):
    if distance not in {"tanimoto", "euclidean", "cosine"}:
        raise ValueError(
            (
                f"Invalid distance metric '{distance}'. Choose between "
                "'tanimoto', 'euclidean', 'cosine'"
            )
        )

    if distance == "tanimoto":
        fingerprints_rdkit = embeddings_to_rdkit(embeddings)
        distances = tanimoto_distance_matrix(fingerprints_rdkit)
    elif distance == "euclidean":
        distances = euclidean_distance_matrix(embeddings)
    elif distance == "cosine":
        distances = cosine_distance_matrix(embeddings)

    clusters = Butina.ClusterData(
        data=distances, nPts=len(embeddings), distThresh=threshold, isDistData=True
    )

    return reformat_clusters(clusters, n_observations=len(embeddings))


def reformat_clusters(clusters, n_observations):
    """Reformat clusters to match scikit-learn format"""
    cluster_assignment = np.zeros(n_observations, dtype=np.int32)
    for cluster_id, cluster in enumerate(clusters):
        for observation in cluster:
            cluster_assignment[observation] = cluster_id
    return cluster_assignment


def tanimoto_distance_matrix(fingerprints, n_cores=N_JOBS):
    distances = Parallel(n_jobs=n_cores, prefer="threads")(
        delayed(DataStructs.BulkTanimotoSimilarity)(
            fingerprints[i], fingerprints[:i], returnDistance=True
        )
        for i in range(1, len(fingerprints))
    )
    return np.concatenate(distances, axis=None)


def euclidean_distance_matrix(embeddings, n_cores=N_JOBS):
    return compute_distances_parallel(embeddings, euclidean_distance, n_cores)


def cosine_distance_matrix(embeddings, n_cores=N_JOBS):
    return compute_distances_parallel(embeddings, cosine_distance, n_cores)


def euclidean_distance(embeddings, i: int, j: int) -> list:
    distance = Butina.EuclideanDist(embeddings[i, :], embeddings[j, :])
    distance = (distance**2).sum() ** 0.5
    return np.expand_dims(np.array(distance), axis=0)


def cosine_distance(embeddings, i: int, j: int) -> np.ndarray:
    A = embeddings[i, :]
    B = embeddings[j, :]
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cosine_similarity = dot_product / (norm_A * norm_B)
    cosine_distance = 1 - cosine_similarity
    return np.expand_dims(np.array(cosine_distance), axis=0)


def compute_distances_parallel(embeddings, compute_distance: callable, n_jobs: int):
    """Computes pairwise distances between embeddings in parallel.
    Adapted from rdkit documentation"""
    distances = Parallel(n_jobs=n_jobs)(
        delayed(compute_distance)(embeddings, i, j)
        for i in range(embeddings.shape[0])
        for j in range(i)
    )
    return np.concatenate(distances)


def embeddings_to_rdkit(embeddings: np.ndarray) -> list:
    fingerprints = []
    for emb in embeddings:
        fingerprint = ndarray_to_binary_string(emb)
        fingerprints.append(DataStructs.CreateFromBitString(fingerprint))

    return fingerprints


def ndarray_to_binary_string(array: np.ndarray) -> str:
    if not len(array) or not is_valid_fingerprint(array):
        raise ValueError(
            "Invalid fingerprint array. Expected binary Morgan fingerprint with 0s and 1s."
        )
    return "".join(array.astype(str).tolist())


def is_valid_fingerprint(fingerprint: np.ndarray) -> bool:
    return np.all(np.isin(fingerprint, [0, 1]))
