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


# ---------------------------------------------------------------- per-dataset settings, sparse data
import json  # noqa: E402

from knee_analogs.core import DatasetSpec, eol_cycle, near_duplicates, reached_eol  # noqa: E402
from knee_analogs import evaluate as ev  # noqa: E402


def sparse_cell(knee: int, seed: int, nominal: float = 3.0, every: int = 50, life: int | None = None,
                condition: str = "c0") -> pd.DataFrame:
    """Check-up capacity every ``every`` cycles on a 3 Ah cell: slow linear fade, then a knee."""
    rng = np.random.default_rng(seed)
    c = np.arange(0, (life or knee + 600) + 1, every, dtype=float)
    q = nominal * (1.0 - 0.00004 * c - 0.0004 * np.maximum(0, c - knee))
    q += rng.normal(0, 0.0005 * nominal, len(c))
    return pd.DataFrame({"cycle": c, "QD": q, "condition": condition})


def write_sparse(folder, knees, **kw):
    for i, k in enumerate(knees):
        sparse_cell(k, i, condition=f"c{i % 3}", **kw).to_csv(folder / f"s{i}.csv", index=False)
    DatasetSpec(name="synthetic_sparse", nominal_ah=3.0, eol_fraction=0.8, min_start_fraction=None, sparse=True,
                early_cycles=(300, 500), main_n=300, checkpoints=(300, 500, 700),
                snapshot_start=100, state_window=150, warning_window=150).to_json(str(folder))


def test_dataset_json_roundtrip_and_unknown_keys(tmp_path):
    spec = DatasetSpec(name="x", nominal_ah=2.5, eol_fraction=0.7, sparse=True, checkpoints=(10, 20))
    spec.to_json(str(tmp_path))
    assert DatasetSpec.from_folder(str(tmp_path)) == spec
    (tmp_path / "dataset.json").write_text(json.dumps({"nominal": 2.5}))
    try:
        DatasetSpec.from_folder(str(tmp_path))
        raise AssertionError("unknown key accepted")
    except ValueError:
        pass
    assert DatasetSpec.from_folder(str(tmp_path / "missing")) == DatasetSpec()  # Severson defaults


def test_sparse_knee_and_interpolated_eol(tmp_path):
    knees = [800, 1000, 1200, 1500]
    write_sparse(tmp_path, knees)
    cells = load_cells(str(tmp_path))
    assert len(cells) == 4
    for k, (name, d) in zip(knees, cells.items()):
        assert (d["QDs"] == d["QD"]).all()                  # sparse data is not smoothed
        assert abs(knee_point(d)[0] - k) < 60, (name, knee_point(d)[0], k)
        assert reached_eol(d)
        # EOL lies between the two check-ups that bracket 80% of nominal, not on one of them
        e = eol_cycle(d)
        assert d["QDs"][d["cycle"] < e].min() >= 2.4 > d["QDs"][d["cycle"] > e].max()


def test_cell_that_never_reaches_eol_is_flagged(tmp_path):
    sparse_cell(5000, 0, life=1000).to_csv(tmp_path / "s0.csv", index=False)
    DatasetSpec(nominal_ah=3.0, sparse=True, min_start_fraction=None).to_json(str(tmp_path))
    d = load_cells(str(tmp_path))["s0"]
    assert not reached_eol(d) and eol_cycle(d) == d["cycle"].iloc[-1]


def test_near_duplicates_found(tmp_path):
    write_cells(tmp_path, [300, 350])
    d = pd.read_csv(tmp_path / "cell0_data.csv")
    d["QD"] += 1e-5                      # same cell re-exported: bytes differ, curve does not
    d.to_csv(tmp_path / "cell0_reexport_data.csv", index=False)
    pairs = near_duplicates(load_cells(str(tmp_path)))
    assert [(a, b) for a, b, _ in pairs] == [("cell0", "cell0_reexport")]


def test_leave_condition_out_never_uses_the_query_condition(tmp_path):
    knees = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 950]
    write_sparse(tmp_path, knees)
    cells = load_cells(str(tmp_path))
    T, C = ev.cell_table(cells, 300)
    P = ev.loco_knee(T, C, leave_out="condition")
    for cell, row in P.iterrows():
        nbs = row["analogs_features_neighbors"].split(",")
        assert all(T.loc[nb, "condition"] != T.loc[cell, "condition"] for nb in nbs)
    S = ev.snapshots(cells)
    assert set(S["cycle"]) <= set(np.concatenate([d["cycle"] for d in cells.values()]))  # check-up cycles only
    W = ev.loco_warning(S, checkpoints=(300, 500), window=150, sparse=True, leave_out="condition")
    assert len(W) and W["analogs"].notna().all()


def test_statistics_helpers():
    actual = pd.Series([100, 300, 310, 320, 330, 340, 350, 900.0])
    typ = ev.main_group(actual)
    assert not typ.iloc[0] and not typ.iloc[-1] and typ.iloc[1:-1].all()
    P = pd.DataFrame({"actual": actual, "a": actual + 5, "b": actual + 20})
    bd = ev.bootstrap_mae_diff(P, "a", "b")
    assert bd["diff"] == -15 and bd["ci_lo"] <= -15 <= bd["ci_hi"]
    assert ev.rank_corr(P, "a", typ) == 1.0


def test_run_study_end_to_end_dense_and_sparse(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dense, sparse = tmp_path / "dense_synth", tmp_path / "sparse_synth"
    dense.mkdir(); sparse.mkdir()
    write_cells(dense, [260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 300, 350])
    write_sparse(sparse, [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 950, 1050])
    for folder in (dense, sparse):
        name = f"_pytest_{folder.name}"
        r = subprocess.run([sys.executable, str(root / "run_study.py"), "--data", str(folder), "--name", name],
                           capture_output=True, text=True, cwd=root)
        out = root / "reports" / name
        try:
            assert r.returncode == 0, r.stderr[-2000:]
            assert (out / "testA_detail.csv").exists() and (out / "figs" / "knees_detected.png").exists()
        finally:
            import shutil
            shutil.rmtree(out, ignore_errors=True)
