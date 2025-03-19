from pathlib import Path
from dataclasses import dataclass

REPO_ROOT = Path(__file__).parent.parent
DATA_PUBLICATION = REPO_ROOT / "data_publication"
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"


@dataclass
class L1000:
    publication = DATA_PUBLICATION / "L1000.csv"


@dataclass
class CellPainting:
    publication = DATA_PUBLICATION / "CP.csv"


@dataclass
class CDDD:
    publication = DATA_PUBLICATION / "CDDD.csv"


@dataclass
class Experiments:
    model_comparison = RESULTS / "model_comparison"
    fusion_comparison = RESULTS / "fusion_comparison"
    increase_observations = RESULTS / "increase_observations"
    complement_modalities = RESULTS / "complement_modalities"
    stacking_comparison = RESULTS / "stacking_comparison"
    dili_prediction = RESULTS / "dili_prediction"
