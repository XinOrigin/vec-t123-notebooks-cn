"""Generate and execute bilingual T1/T2/T3 tutorial notebooks deterministically."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def _md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def _code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_t1_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(
            """
# Task 1: Temporal scRNA-seq Extrapolation (E8.5 → E9.5) | 时序单细胞外推实战

**Bilingual Chinese + English Walkthrough (`vec-t123-notebooks-cn`)**

- **目标 / Goal**: Given earlier-stage embryonic single-cell expression (`E8.5`), predict future-stage transcriptomes (`E9.5`) matching `var_names` order and `log1p` non-negative normalized expression.
- **无泄漏本地验证 / Leakage-Free Setup**: Split the pseudo-target stage cleanly into `train_fit` (used only to estimate developmental shift `Δ`) and `eval_holdout` (used strictly for metrics and visualization).
- **双模式运行 / Dual Mode**: Automatically uses `../veckit/data/` official sample `.h5ad` files if available, or self-contained synthetic data anywhere (Jupyter / Colab / CI).
"""
        ),
        _code(
            """
import json, sys, tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

for candidate in (Path.cwd(), Path.cwd().parent):
    if (candidate / "vec_nb_helpers.py").is_file() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from vec_nb_helpers import (
    _dense_float,
    check_submission_contract,
    load_task_pair,
    predict_pseudobulk_shift,
    score_with_veckit_optional,
)

ref, target_full, source_tag = load_task_pair("T1")
print(f"Loaded T1 pair from: {source_tag}")
print(f"Reference shape: {ref.shape} | Target shape: {target_full.shape}")
print("obs columns:", list(ref.obs.columns))
"""
        ),
        _md(
            """
## 1. 无泄漏切分与双基线对比 / Leakage-Free Split & Baseline Comparison

对比两种最基础却差异巨大的基线：
1. **`copy_last`**: 直接复制 `E8.5` 作为 `E9.5` 预测（`de_score` 往往接近 0 分锚点）。
2. **`fit_split_shift`**: 只在 `eval_target` 之外的 `fit_target` 半折上估计基因均值漂移，并在独立留出半折 `eval_target` 上验证与可视化。
"""
        ),
        _code(
            """
rng = np.random.default_rng(42)
perm = rng.permutation(target_full.n_obs)
half = target_full.n_obs // 2
fit_target = target_full[perm[:half]].copy()
eval_target = target_full[perm[half:]].copy()

x_ref = _dense_float(ref)
x_fit = _dense_float(fit_target)
x_eval = _dense_float(eval_target)

delta_fit = x_fit.mean(axis=0) - x_ref.mean(axis=0)
pred_copy = predict_pseudobulk_shift(ref, np.zeros_like(delta_fit), noise_std=0.0)
pred_shift = predict_pseudobulk_shift(ref, delta_fit, noise_std=0.0)

true_de = x_eval.mean(axis=0) - x_ref.mean(axis=0)
copy_de = _dense_float(pred_copy).mean(axis=0) - x_ref.mean(axis=0)
shift_de = _dense_float(pred_shift).mean(axis=0) - x_ref.mean(axis=0)

def pearson(a, b):
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    return float(np.dot(a_c, b_c) / denom) if denom > 0 else 0.0

print(f"Holdout DE Pearson(copy_last) : {pearson(true_de, copy_de):.4f}")
print(f"Holdout DE Pearson(shift_fit) : {pearson(true_de, shift_de):.4f}")
print("Contract check (pred_shift):", check_submission_contract(pred_shift, ref, require_spatial=False))
"""
        ),
        _md(
            """
## 2. 基因差分漂移可视化 & `veckit` 本地评分 / DE Shift Plot & Local `veckit` Scoring
"""
        ),
        _code(
            """
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: Top differentially expressed genes scatter (True eval delta vs Predicted delta)
top_idx = np.argsort(np.abs(true_de))[-250:]
axes[0].scatter(true_de[top_idx], shift_de[top_idx], s=14, alpha=0.65, color="#2563eb", label="shift_fit")
lim = max(0.3, float(np.max(np.abs(true_de[top_idx])))) * 1.1
axes[0].plot([-lim, lim], [-lim, lim], "--", color="#dc2626", lw=1.2, label="y = x")
axes[0].set_title("T1 Top-250 DE Genes: True vs Predicted Shift")
axes[0].set_xlabel("Holdout True Δ (eval - ref)")
axes[0].set_ylabel("Predicted Δ (pred - ref)")
axes[0].legend()
axes[0].grid(alpha=0.25)

# Panel B: Celltype proportion drift between reference and eval stage
ct_ref = ref.obs["celltype"].astype(str).value_counts(normalize=True)
ct_eval = eval_target.obs["celltype"].astype(str).value_counts(normalize=True)
cts = sorted(set(ct_ref.index) | set(ct_eval.index))[:8]
x_pos = np.arange(len(cts))
axes[1].bar(x_pos - 0.18, [ct_ref.get(c, 0.0) for c in cts], width=0.36, label="E8.5 ref", color="#64748b")
axes[1].bar(x_pos + 0.18, [ct_eval.get(c, 0.0) for c in cts], width=0.36, label="E9.5 eval", color="#059669")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(cts, rotation=28, ha="right", fontsize=8)
axes[1].set_title("Cell-Type Proportion Shift (Top Categories)")
axes[1].set_ylabel("Fraction")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

with tempfile.TemporaryDirectory() as tmp:
    tmp_p = Path(tmp)
    p_ref, p_tgt, p_copy, p_shift = tmp_p / "ref.h5ad", tmp_p / "tgt.h5ad", tmp_p / "copy.h5ad", tmp_p / "shift.h5ad"
    ref.write_h5ad(p_ref)
    eval_target.write_h5ad(p_tgt)
    pred_copy.write_h5ad(p_copy)
    pred_shift.write_h5ad(p_shift)
    s_copy = score_with_veckit_optional("T1", p_copy, p_tgt, p_ref)
    s_shift = score_with_veckit_optional("T1", p_shift, p_tgt, p_ref)
    if s_copy and s_shift:
        print("veckit copy_last:", json.dumps(s_copy["metrics"], indent=2))
        print("veckit shift_fit:", json.dumps(s_shift["metrics"], indent=2))
    else:
        print("veckit not installed in env — contract & holdout Pearson checks completed offline.")
"""
        ),
    ]
    return nb


def build_t2_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(
            """
# Task 2: 3D Spatial Embryo & Heart Morphogenesis | 3D 空间形态发生建模实战

**为什么只预测表达、不改变 `obsm["spatial_3D"]` 会在 T2/T3 丢分？**

In Task 2 (Heart E9.25 → E9.5 extrapolation & Whole-Embryo interpolation), evaluation scores **both** gene expression (`de_score`, `celltype_de`, `energy_distance`) and **3D spatial geometry** (`d2_shape`, `sliced_wasserstein`, `occupancy_dice`, `scale_log_ratio`). Leaving `obsm["spatial_3D"]` static ignores organ expansion, axis rotation, and translation.
"""
        ),
        _code(
            """
import json, sys, tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

for candidate in (Path.cwd(), Path.cwd().parent):
    if (candidate / "vec_nb_helpers.py").is_file() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from vec_nb_helpers import (
    SPATIAL_KEY,
    check_submission_contract,
    load_task_pair,
    predict_joint_3d_morphogenesis,
    predict_pseudobulk_shift,
    score_with_veckit_optional,
)

ref, target_full, source_tag = load_task_pair("T2")
print(f"Loaded T2 pair from: {source_tag} | ref={ref.shape}, target={target_full.shape}")
print("spatial_3D shape:", ref.obsm[SPATIAL_KEY].shape, "dtype:", ref.obsm[SPATIAL_KEY].dtype)
"""
        ),
        _md(
            """
## 1. 表达迁移 vs 联合 3D 协方差形态映射 / Static Spatial vs Joint 3D Transform

同样采用严格的 `fit_target` / `eval_target` 不相交划分，避免信息泄漏，并比较 3D 半径尺度比（`RMS radius`）与 3D 空间投影变化。
"""
        ),
        _code(
            """
rng = np.random.default_rng(7)
perm = rng.permutation(target_full.n_obs)
half = target_full.n_obs // 2
fit_target = target_full[perm[:half]].copy()
eval_target = target_full[perm[half:]].copy()

pred_static_spatial = predict_pseudobulk_shift(ref, np.zeros(ref.n_vars, dtype=np.float32), noise_std=0.0)
pred_joint_3d = predict_joint_3d_morphogenesis(ref, fit_target, shrink=0.80)

def rms_radius(adata):
    coords = np.asarray(adata.obsm[SPATIAL_KEY], dtype=np.float64)
    centered = coords - coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))

r_eval = rms_radius(eval_target)
r_static = rms_radius(pred_static_spatial)
r_joint = rms_radius(pred_joint_3d)
print(f"Eval RMS 3D radius          : {r_eval:.3f}")
print(f"Static-spatial RMS 3D radius: {r_static:.3f} (scale ratio={r_static/r_eval:.3f})")
print(f"Joint-3D RMS 3D radius      : {r_joint:.3f} (scale ratio={r_joint/r_eval:.3f})")
print("Contract check (joint_3d):", check_submission_contract(pred_joint_3d, ref, require_spatial=True))
"""
        ),
        _code(
            """
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
s_static = np.asarray(pred_static_spatial.obsm[SPATIAL_KEY])
s_joint = np.asarray(pred_joint_3d.obsm[SPATIAL_KEY])
s_eval = np.asarray(eval_target.obsm[SPATIAL_KEY])

for ax, pts, title, col in [
    (axes[0], s_static, "Static Spatial (E9.25 copy)", "#64748b"),
    (axes[1], s_joint, "Joint 3D Morphogenesis Pred", "#2563eb"),
    (axes[2], s_eval, "Holdout Eval Target (E9.5)", "#059669"),
]:
    ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.7, color=col)
    ax.set_title(title)
    ax.set_xlabel("spatial_3D[:, 0]")
    ax.set_ylabel("spatial_3D[:, 1]")
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()

with tempfile.TemporaryDirectory() as tmp:
    tmp_p = Path(tmp)
    p_ref, p_tgt = tmp_p / "ref.h5ad", tmp_p / "tgt.h5ad"
    p_stat, p_jnt = tmp_p / "static.h5ad", tmp_p / "joint.h5ad"
    ref.write_h5ad(p_ref)
    eval_target.write_h5ad(p_tgt)
    pred_static_spatial.write_h5ad(p_stat)
    pred_joint_3d.write_h5ad(p_jnt)
    s_stat = score_with_veckit_optional("T2", p_stat, p_tgt, p_ref, setting="heart")
    s_jnt = score_with_veckit_optional("T2", p_jnt, p_tgt, p_ref, setting="heart")
    if s_stat and s_jnt:
        print("veckit static_spatial:", json.dumps(s_stat["metrics"], indent=2))
        print("veckit joint_3d      :", json.dumps(s_jnt["metrics"], indent=2))
"""
        ),
    ]
    return nb


def build_t3_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _md(
            """
# Task 3: Knockout Perturbation Delta Modeling (`WT → Mab21l2_KO`) | 基因敲除扰动 Delta 预测实战

**T3 核心难点 / Core Challenge of Task 3**:

Task 3 requires predicting both **transcriptional shift** and **3D morphogenesis changes** under genetic knockouts (`Mab21l2`, `Gata4`, `Ctnnb1`). Because cell types respond unevenly to perturbations—and rare populations suffer high variance—we demonstrate a **shrinkage cell-type perturbation delta estimator** (`shrink * Δ_celltype + (1 - shrink) * Δ_global`) coupled with 3D spatial scaling, evaluated on an isolated holdout fold.
"""
        ),
        _code(
            """
import json, sys, tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

for candidate in (Path.cwd(), Path.cwd().parent):
    if (candidate / "vec_nb_helpers.py").is_file() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from vec_nb_helpers import (
    SPATIAL_KEY,
    _dense_float,
    check_submission_contract,
    load_task_pair,
    predict_celltype_perturbation_delta,
    predict_pseudobulk_shift,
    score_with_veckit_optional,
)

wt, ko_full, source_tag = load_task_pair("T3")
print(f"Loaded T3 pair from: {source_tag} | WT={wt.shape}, KO={ko_full.shape}")
print("T3 genes head:", list(wt.var_names[:6]))
"""
        ),
        _md(
            """
## 1. `wt_identity` 基线 vs 细胞类型收缩扰动 Delta / Identity vs Shrinkage Cell-Type Delta

这里严格将 `ko_full` 拆为 `ko_fit` 与 `ko_eval`，用 `ko_fit` 拟合每个细胞类型的扰动响应向量与 3D 空间尺度变换，并在从未参与拟合的 `ko_eval` 上对比方向一致性与显著 Marker 误差。
"""
        ),
        _code(
            """
rng = np.random.default_rng(13)
perm = rng.permutation(ko_full.n_obs)
half = ko_full.n_obs // 2
ko_fit = ko_full[perm[:half]].copy()
ko_eval = ko_full[perm[half:]].copy()

pred_wt_id = predict_pseudobulk_shift(wt, np.zeros(wt.n_vars, dtype=np.float32), noise_std=0.0)
pred_ct_delta = predict_celltype_perturbation_delta(wt, ko_fit, obs_key="celltype", shrink=0.75, min_cells=5)

x_wt = _dense_float(wt)
x_eval = _dense_float(ko_eval)
x_pred = _dense_float(pred_ct_delta)

true_delta = x_eval.mean(axis=0) - x_wt.mean(axis=0)
pred_delta = x_pred.mean(axis=0) - x_wt.mean(axis=0)

sign_acc = float(np.mean(np.sign(true_delta) == np.sign(pred_delta)))
mae_wt_id = float(np.mean(np.abs(true_delta)))
mae_ct = float(np.mean(np.abs(true_delta - pred_delta)))
print(f"Mean |DE| error (wt_identity)       : {mae_wt_id:.4f}")
print(f"Mean |DE| error (celltype_delta_pred): {mae_ct:.4f}")
print(f"DE sign agreement vs eval knockout   : {sign_acc:.3%}")
print("Contract check (T3 pred):", check_submission_contract(pred_ct_delta, wt, require_spatial=True))
"""
        ),
        _code(
            """
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

# Panel A: Top-12 perturbed genes comparison (True delta vs Predicted delta)
top12 = np.argsort(np.abs(true_delta))[-12:][::-1]
y = np.arange(len(top12))
names = [str(wt.var_names[i]) for i in top12]
axes[0].barh(y - 0.18, true_delta[top12], height=0.36, label="Eval True KO Δ", color="#059669")
axes[0].barh(y + 0.18, pred_delta[top12], height=0.36, label="Predicted Δ", color="#2563eb")
axes[0].axvline(0.0, color="#0f172a", lw=0.8)
axes[0].set_yticks(y)
axes[0].set_yticklabels(names, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_title("Top-12 Perturbed Genes (Δ log1p vs WT)")
axes[0].legend()
axes[0].grid(axis="x", alpha=0.25)

# Panel B: 3D spatial comparison WT vs KO eval vs Predicted
s_wt = np.asarray(wt.obsm[SPATIAL_KEY])
s_eval = np.asarray(ko_eval.obsm[SPATIAL_KEY])
s_pred = np.asarray(pred_ct_delta.obsm[SPATIAL_KEY])
axes[1].scatter(s_wt[:, 0], s_wt[:, 1], s=14, alpha=0.35, label="WT", color="#94a3b8")
axes[1].scatter(s_eval[:, 0], s_eval[:, 1], s=18, alpha=0.65, label="Eval KO", color="#059669")
axes[1].scatter(s_pred[:, 0], s_pred[:, 1], s=18, alpha=0.65, marker="^", label="Pred KO", color="#2563eb")
axes[1].set_title("T3 Spatial Morphogenesis Projection (X vs Y)")
axes[1].set_xlabel("spatial_3D[:, 0]")
axes[1].set_ylabel("spatial_3D[:, 1]")
axes[1].legend()
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.show()

with tempfile.TemporaryDirectory() as tmp:
    tmp_p = Path(tmp)
    p_wt, p_tgt = tmp_p / "wt.h5ad", tmp_p / "tgt.h5ad"
    p_id, p_ct = tmp_p / "wt_id.h5ad", tmp_p / "ct_delta.h5ad"
    wt.write_h5ad(p_wt)
    ko_eval.write_h5ad(p_tgt)
    pred_wt_id.write_h5ad(p_id)
    pred_ct_delta.write_h5ad(p_ct)
    s_id = score_with_veckit_optional("T3", p_id, p_tgt, p_wt)
    s_ct = score_with_veckit_optional("T3", p_ct, p_tgt, p_wt)
    if s_id and s_ct:
        print("veckit wt_identity :", json.dumps(s_id["metrics"], indent=2))
        print("veckit ct_delta    :", json.dumps(s_ct["metrics"], indent=2))
"""
        ),
    ]
    return nb


def execute_notebook(path: Path) -> None:
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    client.execute()
    nbf.write(nb, path)


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("01_t1_temporal_extrapolation.ipynb", build_t1_notebook()),
        ("02_t2_spatial_3d_morphogenesis.ipynb", build_t2_notebook()),
        ("03_t3_perturbation_knockout_delta.ipynb", build_t3_notebook()),
    ]
    for fname, nb in specs:
        out = NB_DIR / fname
        nbf.write(nb, out)
        execute_notebook(out)
        print(f"Executed cleanly: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
