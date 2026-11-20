"""Named training runs: which clause types, where the data, adapter and results live.

  top6         the course recipe: LEDGAR's 6 most common clause types (milestones 1-6)
  procurement  14 clause types chosen after reading the Indian public records; this is
               the adapter `python -m app.cli tag` uses by default (models/adapter.pt)
"""
from dataclasses import dataclass
from pathlib import Path

from app.data import PROCUREMENT_LABELS


@dataclass(frozen=True)
class Run:
    name: str
    data_dir: Path
    adapter: Path
    out: Path
    top_k: int = 6
    labels: tuple[str, ...] | None = None    # explicit label list; None = the top_k most common


RUNS = {
    "top6": Run("top6", Path("data/ledgar"), Path("models/adapter_top6.pt"), Path("outputs/top6"), top_k=6),
    "procurement": Run("procurement", Path("data/ledgar_procurement"), Path("models/adapter.pt"),
                       Path("outputs/procurement"), labels=tuple(PROCUREMENT_LABELS)),
}
DEFAULT_RUN = "top6"


def get_run(name: str) -> Run:
    if name not in RUNS:
        raise ValueError(f"unknown run {name!r}; choose one of {sorted(RUNS)}")
    return RUNS[name]
