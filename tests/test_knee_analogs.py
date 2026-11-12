"""Synthetic-cell tests: knee detection, data loading safeguards, and no self-matching."""

import numpy as np
import pandas as pd

from knee_analogs.core import knee_point, load_cells
from knee_analogs.evaluate import cell_table, loco_knee, loco_warning, snapshots, warning_summary


def synthetic_cell(knee: int, seed: int, n_after: int = 160, start: float = 1.08) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = np.arange(1, knee + n_after + 1, dtype=float)
    q = start - 0.00012 * c - 0.0012 * np.maximum(0, c - knee) - 4e-6 * np.maximum(0, c - knee) ** 2
    q += rng.normal(0, 0.0007, len(c))
    pol = ["4C(20%)-5C", "5C(40%)-4C", "6C(30%)-4.5C"][seed % 3]
    return pd.DataFrame({"cycle": c, "QD": q, "IR": 0.017 + 1e-6 * c, "Tavg": 31 + 0.001 * c,
                         "chargetime": 10.0, "charge_policy": pol})


def write_cells(folder, knees):
    for i, k in enumerate(knees):
        synthetic_cell(k, i).to_csv(folder / f"cell{i}_data.csv", index=False)


def test_knee_point_recovers_breakpoint():
    for k in (250, 350, 450):
        d = load_cells_from_frame(synthetic_cell(k, seed=k))
        est = knee_point(d)[0]
        assert abs(est - k) < 40, (k, est)


def load_cells_from_frame(df):
    from knee_analogs.core import _clean

    d = _clean(df)
    d.attrs["policy"] = df["charge_policy"].iloc[0]
    return d


def test_load_cells_drops_duplicates_and_incomplete_cells(tmp_path):
    write_cells(tmp_path, [300, 350])
    (tmp_path / "copy_of_cell0_data.csv").write_bytes((tmp_path / "cell0_data.csv").read_bytes())
    synthetic_cell(320, 9, start=1.0).to_csv(tmp_path / "continued_cell_data.csv", index=False)
    cells = load_cells(str(tmp_path))
    assert sorted(cells) == ["cell0", "cell1"]


def test_leave_one_cell_out_never_matches_itself(tmp_path):
    knees = [260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 300, 350]
    write_cells(tmp_path, knees)
    cells = load_cells(str(tmp_path))
    T, C = cell_table(cells, 100)
    P = loco_knee(T, C)
    for cell, row in P.iterrows():
        assert cell not in row["analogs_features_neighbors"].split(",")
        assert cell not in row["analogs_curve_neighbors"].split(",")
    assert np.isfinite(P["analogs_features"]).all()
    W = loco_warning(snapshots(cells), checkpoints=(100, 200))
    s = warning_summary(W) if (W["actual"] <= 50).any() else {"AUC_soon_analogs": 0.5}
    assert 0 <= s["AUC_soon_analogs"] <= 1
