"""Verify every tutorial notebook executes end-to-end with non-empty outputs and contract checks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

from vec_nb_helpers import (
    check_submission_contract,
    make_synthetic_t1_pair,
    make_synthetic_t2_pair,
    make_synthetic_t3_pair,
    predict_celltype_perturbation_delta,
    predict_joint_3d_morphogenesis,
)

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def test_helper_baselines_produce_valid_contracts():
    r1, _ = make_synthetic_t1_pair()
    assert check_submission_contract(r1, r1, require_spatial=False)["ok"]

    r2, t2 = make_synthetic_t2_pair()
    p2 = predict_joint_3d_morphogenesis(r2, t2, shrink=0.75)
    assert check_submission_contract(p2, r2, require_spatial=True)["ok"]

    wt, ko = make_synthetic_t3_pair()
    p3 = predict_celltype_perturbation_delta(wt, ko, shrink=0.7, min_cells=4)
    assert check_submission_contract(p3, wt, require_spatial=True)["ok"]


def test_all_three_notebooks_have_saved_outputs_and_execute_cleanly(tmp_path: Path):
    expected = [
        "01_t1_temporal_extrapolation.ipynb",
        "02_t2_spatial_3d_morphogenesis.ipynb",
        "03_t3_perturbation_knockout_delta.ipynb",
    ]
    for name in expected:
        nb_path = NB_DIR / name
        assert nb_path.is_file(), f"Missing notebook: {nb_path}"
        nb = nbf.read(nb_path, as_version=4)

        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert code_cells, f"{name} has no code cells"
        assert all(c.outputs for c in code_cells), f"{name} has unexecuted code cells"
        assert any(
            "image/png" in out.get("data", {})
            for c in code_cells
            for out in c.outputs
        ), f"{name} is missing embedded matplotlib plot output"

        # Re-execute a fresh copy in headless mode to guarantee reproducibility
        client = NotebookClient(
            nb,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(NB_DIR)}},
        )
        client.execute()
