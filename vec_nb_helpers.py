"""Shared deterministic helpers for Virtual Embryo Challenge T1/T2/T3 bilingual notebooks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

SPATIAL_KEY = "spatial_3D"


def _dense_float(adata: ad.AnnData) -> np.ndarray:
    """Return dense float32 expression matrix without mutating source AnnData."""
    mat = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    return np.asarray(mat, dtype=np.float32)


def find_veckit_dir(start: Path | None = None) -> Path | None:
    """Locate a local `veckit` checkout containing `veckit.py` and `data/`."""
    base = (start or Path.cwd()).resolve()
    for candidate in (base / "veckit", base.parent / "veckit", base.parent.parent / "veckit"):
        if (candidate / "veckit.py").is_file() and (candidate / "data").is_dir():
            return candidate
    return None


def make_synthetic_t1_pair(n_cells: int = 120, n_genes: int = 400, seed: int = 42) -> tuple[ad.AnnData, ad.AnnData]:
    """Generate realistic synthetic T1 (E8.5 -> E9.5) log1p single-cell AnnData files."""
    rng = np.random.default_rng(seed)
    celltypes = np.array(["Endoderm", "Mesoderm", "Neural_Crest", "Cardiomyocytes"])
    genes = [f"Gene_{i:04d}" for i in range(n_genes)]

    obs_ref = {"celltype": rng.choice(celltypes, size=n_cells, p=[0.25, 0.30, 0.25, 0.20])}
    obs_tgt = {"celltype": rng.choice(celltypes, size=n_cells, p=[0.22, 0.28, 0.25, 0.25])}

    # Stage shift vector: positive + negative developmental trajectory changes
    stage_shift = rng.normal(0.0, 0.35, size=n_genes).astype(np.float32)
    ct_offsets = {
        ct: rng.normal(0.0, 0.4, size=n_genes).astype(np.float32) for ct in celltypes
    }

    def sample_expr(cts: np.ndarray, extra_shift: np.ndarray) -> np.ndarray:
        base = rng.uniform(0.2, 2.0, size=(1, n_genes)).astype(np.float32)
        offsets = np.vstack([ct_offsets[c] for c in cts])
        noise = rng.normal(0.0, 0.3, size=(len(cts), n_genes)).astype(np.float32)
        return np.clip(base + offsets + extra_shift + noise, 0.0, None).astype(np.float32)

    ref = ad.AnnData(X=sample_expr(obs_ref["celltype"], np.zeros(n_genes, dtype=np.float32)), obs=obs_ref)
    tgt = ad.AnnData(X=sample_expr(obs_tgt["celltype"], stage_shift), obs=obs_tgt)
    ref.var_names = genes
    tgt.var_names = genes
    return ref, tgt


def make_synthetic_t2_pair(n_cells: int = 120, n_genes: int = 500, seed: int = 7) -> tuple[ad.AnnData, ad.AnnData]:
    """Generate synthetic T2 (E9.25 -> E9.5) expression + 3D spatial AnnData pair."""
    rng = np.random.default_rng(seed)
    celltypes = np.array(["Atrial_CM", "Ventricular_CM", "Endocardial", "Epicardial"])
    genes = [f"Merfish_{i:03d}" for i in range(n_genes)]

    ct_ref = rng.choice(celltypes, size=n_cells)
    ct_tgt = rng.choice(celltypes, size=n_cells)

    coords_ref = rng.normal(0.0, 1.0, size=(n_cells, 3)).astype(np.float32)
    # Anisotropic 3D morphogenesis growth + translation
    growth = np.array([1.45, 1.15, 0.90], dtype=np.float32)
    coords_tgt = (rng.normal(0.0, 1.0, size=(n_cells, 3)).astype(np.float32) * growth) + np.array(
        [0.4, -0.2, 0.3], dtype=np.float32
    )

    expr_ref = np.clip(rng.normal(1.2, 0.45, size=(n_cells, n_genes)), 0.0, None).astype(np.float32)
    expr_shift = rng.normal(0.0, 0.25, size=(1, n_genes)).astype(np.float32)
    expr_tgt = np.clip(rng.normal(1.2, 0.45, size=(n_cells, n_genes)) + expr_shift, 0.0, None).astype(np.float32)

    ref = ad.AnnData(X=expr_ref, obs={"celltype": ct_ref}, obsm={SPATIAL_KEY: coords_ref})
    tgt = ad.AnnData(X=expr_tgt, obs={"celltype": ct_tgt}, obsm={SPATIAL_KEY: coords_tgt})
    ref.var_names = genes
    tgt.var_names = genes
    return ref, tgt


def make_synthetic_t3_pair(n_cells: int = 140, n_genes: int = 500, seed: int = 13) -> tuple[ad.AnnData, ad.AnnData]:
    """Generate synthetic T3 (WT -> Knockout) perturbation expression + 3D spatial pair."""
    rng = np.random.default_rng(seed)
    celltypes = np.array(["Eye_Field", "Heart_Tube", "Neural_Plate", "Mesenchyme"])
    markers = ["Mab21l2", "Tbx5", "Hand2", "Nkx2-5", "Isl1"]
    genes = markers + [f"Panel_{i:03d}" for i in range(n_genes - len(markers))]

    ct_wt = rng.choice(celltypes, size=n_cells, p=[0.25, 0.30, 0.25, 0.20])
    ct_ko = rng.choice(celltypes, size=n_cells, p=[0.18, 0.28, 0.29, 0.25])

    ko_effect = {
        "Eye_Field": rng.normal(0.0, 0.30, size=n_genes).astype(np.float32),
        "Heart_Tube": rng.normal(0.0, 0.25, size=n_genes).astype(np.float32),
        "Neural_Plate": rng.normal(0.0, 0.15, size=n_genes).astype(np.float32),
        "Mesenchyme": rng.normal(0.0, 0.15, size=n_genes).astype(np.float32),
    }
    # Knockout strongly suppresses Mab21l2 (gene index 0) and alters cardiac markers
    for ct in ko_effect:
        ko_effect[ct][0] = -1.25
        ko_effect[ct][1] = -0.55 if ct in ("Eye_Field", "Heart_Tube") else -0.15
        ko_effect[ct][2] = 0.40 if ct == "Heart_Tube" else 0.10

    x_wt = np.clip(rng.normal(1.6, 0.4, size=(n_cells, n_genes)), 0.0, None).astype(np.float32)
    delta_mat = np.vstack([ko_effect[c] for c in ct_ko])
    x_ko = np.clip(rng.normal(1.6, 0.4, size=(n_cells, n_genes)) + delta_mat, 0.0, None).astype(np.float32)

    s_wt = rng.normal(0.0, 1.0, size=(n_cells, 3)).astype(np.float32)
    # Hypoplastic knockout morphology (slight contraction)
    s_ko = (rng.normal(0.0, 1.0, size=(n_cells, 3)).astype(np.float32) * 0.88) + np.array(
        [-0.15, 0.10, 0.0], dtype=np.float32
    )

    wt = ad.AnnData(X=x_wt, obs={"celltype": ct_wt}, obsm={SPATIAL_KEY: s_wt})
    ko = ad.AnnData(X=x_ko, obs={"celltype": ct_ko}, obsm={SPATIAL_KEY: s_ko})
    wt.var_names = genes
    ko.var_names = genes
    return wt, ko


def load_task_pair(task: str, veckit_dir: Path | None = None) -> tuple[ad.AnnData, ad.AnnData, str]:
    """Load official `veckit/data` samples when present, else clean synthetic fallback."""
    t = task.upper()
    vk = find_veckit_dir() if veckit_dir is None else Path(veckit_dir)
    mapping = {
        "T1": ("sample_8.5.h5ad", "sample_9.5.h5ad"),
        "T2": ("sample_heart_9.25.h5ad", "sample_heart_9.5.h5ad"),
        "T3": ("sample_wt.h5ad", "sample_mab21l2_ko.h5ad"),
    }
    if t not in mapping:
        raise ValueError(f"Unsupported task: {task!r}")

    if vk is not None:
        p1, p2 = (vk / "data" / mapping[t][0]), (vk / "data" / mapping[t][1])
        if p1.is_file() and p2.is_file():
            return ad.read_h5ad(p1), ad.read_h5ad(p2), f"veckit_sample:{vk.name}"

    if t == "T1":
        a, b = make_synthetic_t1_pair()
    elif t == "T2":
        a, b = make_synthetic_t2_pair()
    else:
        a, b = make_synthetic_t3_pair()
    return a, b, "synthetic_offline"


def predict_pseudobulk_shift(source: ad.AnnData, delta: np.ndarray, noise_std: float = 0.05, seed: int = 0) -> ad.AnnData:
    """Apply a mean expression shift `delta` to `source` and clamp to valid non-negative log1p."""
    rng = np.random.default_rng(seed)
    x = _dense_float(source) + np.asarray(delta, dtype=np.float32).reshape(1, -1)
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    x = np.clip(x, 0.0, None).astype(np.float32)
    pred = ad.AnnData(X=x, obs=source.obs.copy(), var=source.var.copy())
    if SPATIAL_KEY in source.obsm:
        pred.obsm[SPATIAL_KEY] = np.asarray(source.obsm[SPATIAL_KEY], dtype=np.float32).copy()
    return pred


def predict_joint_3d_morphogenesis(
    source: ad.AnnData,
    target_for_stats: ad.AnnData,
    shrink: float = 0.75,
) -> ad.AnnData:
    """Jointly transport expression centroid and 3D spatial mean+covariance (shrunk OT-like map)."""
    x_src = _dense_float(source)
    x_tgt = _dense_float(target_for_stats)
    global_delta = (x_tgt.mean(axis=0) - x_src.mean(axis=0)) * float(shrink)
    pred = predict_pseudobulk_shift(source, global_delta, noise_std=0.0, seed=0)

    s_src = np.asarray(source.obsm[SPATIAL_KEY], dtype=np.float64)
    s_tgt = np.asarray(target_for_stats.obsm[SPATIAL_KEY], dtype=np.float64)
    mu_src, mu_tgt = s_src.mean(axis=0), s_tgt.mean(axis=0)

    # Diagonal-stabilized 3D standard deviation scaling in centered coordinates
    std_src = np.maximum(s_src.std(axis=0), 1e-6)
    std_tgt = np.maximum(s_tgt.std(axis=0), 1e-6)
    scale = (1.0 - shrink) * np.ones(3) + shrink * (std_tgt / std_src)
    s_pred = ((s_src - mu_src) * scale + ((1.0 - shrink) * mu_src + shrink * mu_tgt)).astype(np.float32)
    pred.obsm[SPATIAL_KEY] = s_pred
    return pred


def predict_celltype_perturbation_delta(
    wt: ad.AnnData,
    ko_proxy: ad.AnnData,
    obs_key: str = "celltype",
    shrink: float = 0.70,
    min_cells: int = 8,
) -> ad.AnnData:
    """Predict knockout expression & spatial shift using shrinkage cell-type delta estimators."""
    x_wt = _dense_float(wt)
    x_ko = _dense_float(ko_proxy)
    global_delta = x_ko.mean(axis=0) - x_wt.mean(axis=0)

    if obs_key in wt.obs and obs_key in ko_proxy.obs:
        ct_wt = wt.obs[obs_key].astype(str).to_numpy()
        ct_ko = ko_proxy.obs[obs_key].astype(str).to_numpy()
        deltas = np.zeros_like(x_wt)
        for ct in np.unique(ct_wt):
            m_wt = ct_wt == ct
            m_ko = ct_ko == ct
            if m_ko.sum() >= min_cells:
                ct_delta = x_ko[m_ko].mean(axis=0) - x_wt[m_wt].mean(axis=0)
                deltas[m_wt] = shrink * ct_delta + (1.0 - shrink) * global_delta
            else:
                deltas[m_wt] = shrink * global_delta
    else:
        deltas = np.broadcast_to(shrink * global_delta, x_wt.shape).copy()

    x_pred = np.clip(x_wt + deltas, 0.0, None).astype(np.float32)
    pred = ad.AnnData(X=x_pred, obs=wt.obs.copy(), var=wt.var.copy())

    if SPATIAL_KEY in wt.obsm and SPATIAL_KEY in ko_proxy.obsm:
        s_wt = np.asarray(wt.obsm[SPATIAL_KEY], dtype=np.float64)
        s_ko = np.asarray(ko_proxy.obsm[SPATIAL_KEY], dtype=np.float64)
        std_wt = np.maximum(s_wt.std(axis=0), 1e-6)
        std_ko = np.maximum(s_ko.std(axis=0), 1e-6)
        scale = (1.0 - shrink) * np.ones(3) + shrink * (std_ko / std_wt)
        s_pred = ((s_wt - s_wt.mean(axis=0)) * scale + ((1.0 - shrink) * s_wt.mean(axis=0) + shrink * s_ko.mean(axis=0))).astype(
            np.float32
        )
        pred.obsm[SPATIAL_KEY] = s_pred
    return pred


def check_submission_contract(pred: ad.AnnData, reference: ad.AnnData, require_spatial: bool) -> dict[str, Any]:
    """Run local pre-submission sanity checks matching VEC board contracts."""
    x = _dense_float(pred)
    same_order = list(pred.var_names) == list(reference.var_names)
    non_negative = bool(np.all(x >= 0.0))
    all_finite = bool(np.all(np.isfinite(x)))
    spatial_ok = (not require_spatial) or (
        SPATIAL_KEY in pred.obsm
        and np.asarray(pred.obsm[SPATIAL_KEY]).shape == (pred.n_obs, 3)
        and bool(np.all(np.isfinite(pred.obsm[SPATIAL_KEY])))
    )
    ok = bool(same_order and non_negative and all_finite and spatial_ok and pred.n_obs > 0)
    return {
        "ok": ok,
        "n_obs": int(pred.n_obs),
        "n_vars": int(pred.n_vars),
        "var_names_exact_match": same_order,
        "non_negative": non_negative,
        "all_finite": all_finite,
        "spatial_3D_valid": spatial_ok,
    }


def score_with_veckit_optional(
    task: str,
    pred_path: Path,
    target_path: Path,
    reference_or_wt_path: Path,
    veckit_dir: Path | None = None,
    setting: str = "heart",
) -> dict[str, Any] | None:
    """Run official `veckit.score()` if `veckit` is installed or locally present."""
    vk = find_veckit_dir() if veckit_dir is None else Path(veckit_dir)
    if vk is not None and str(vk) not in sys.path:
        sys.path.insert(0, str(vk))
    try:
        mod = importlib.import_module("veckit")
    except ModuleNotFoundError:
        return None

    t = task.upper()
    if t == "T1":
        return mod.score(task="T1", input=pred_path, target=target_path, reference=reference_or_wt_path)
    if t == "T2":
        return mod.score(task="T2", input=pred_path, target=target_path, reference=reference_or_wt_path, setting=setting)
    if t == "T3":
        return mod.score(task="T3", input=pred_path, target=target_path, wt=reference_or_wt_path)
    raise ValueError(f"Unsupported task: {task!r}")
