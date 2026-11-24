# vec-t123-notebooks-cn

> **Executed Bilingual (中文 / English) Jupyter/Colab Walkthrough Suite** for all three tasks of the [Virtual Embryo Challenge (NeurIPS 2026)](https://virtualembryo.ai/challenge): **T1 (Temporal)**, **T2 (3D Spatial)**, and **T3 (Knockout Perturbation)**.

## Why This Exists / 设计初衷

Many community notebooks cover Task 1 or Task 2 on synthetic data only, while **Task 3 (perturbation delta prediction across `WT → Knockout` with 3D morphology shift)** lacks an executed bilingual starter. This repository provides three **already-executed** `.ipynb` notebooks (with saved `veckit` metrics and inline diagnostic plots) that also run cleanly offline without downloading external datasets.

## Included Notebooks / 包含的 Notebook

| Notebook | Task | What It Teaches |
|---|---|---|
| [`notebooks/01_t1_temporal_extrapolation.ipynb`](notebooks/01_t1_temporal_extrapolation.ipynb) | **T1 (E8.5 → E9.5)** | Leakage-free holdout split, `copy_last` vs `shift_fit` DE correlation, cell-type proportion shift, and `veckit` scoring |
| [`notebooks/02_t2_spatial_3d_morphogenesis.ipynb`](notebooks/02_t2_spatial_3d_morphogenesis.ipynb) | **T2 (3D Spatial)** | Why leaving `obsm["spatial_3D"]` unchanged hurts spatial metrics, and how joint 3D mean+covariance scaling aligns embryo radius and layout |
| [`notebooks/03_t3_perturbation_knockout_delta.ipynb`](notebooks/03_t3_perturbation_knockout_delta.ipynb) | **T3 (`WT → KO` Perturbation)** | Shrinkage cell-type delta modeling (`shrink * Δ_ct + (1 - shrink) * Δ_global`), DE sign-agreement checks, top-12 perturbed gene plots, and 3D KO contraction projection |

## Quick Start / 本地或 Colab 一键运行

```bash
git clone https://github.com/XinOrigin/vec-t123-notebooks-cn.git
cd vec-t123-notebooks-cn
pip install -e ".[notebook]"

# Optional: include official offline panel scores via aristoteleo/vec_baselines
pip install git+https://github.com/aristoteleo/vec_baselines.git

# Re-execute all 3 notebooks and verify embedded outputs
python scripts/build_and_execute_notebooks.py
pytest -v
```

## Scope & Disclaimer / 适用范围说明

- Notebooks automatically use official `veckit/data/sample_*.h5ad` files when a local `veckit` checkout is adjacent, and cleanly fall back to `synthetic_offline` fixtures when run standalone (e.g., Colab or CI).
- Scores computed here are **local diagnostics on public sample/synthetic holdouts** and do not substitute for private validation/test board evaluation.
