#!/usr/bin/env python
"""
Comprehensive analysis of gated tiered MORL runs from a W&B CSV export.

Key ideas:
- Use *base* MORL metrics (not blocks / scalar reward) + ave_alive as true
  objectives.
- Treat blocks (primary / fairness / sustain / structural) as *derived*
  metrics, not independent objectives.
- Ignore ave_r for multi-objective evaluation (reward depends on weights).
- Min-max normalize metrics to [0, 1] and build a high-dimensional Pareto
  front in that normalized space.
- Add gating-aware "effective weight" features (alpha_* * w_*).
- For each normalized metric, fit Linear, Ridge, and Lasso regression on all
  weights + effective weights.
- Compute PCA + UMAP on normalized base metric space and plot, labeling
  Pareto-front runs with run_idx.
- Save readable summaries for Pareto runs and regression-based weight
  suggestions.
- Print all key tables to stdout and save them as red–yellow–green table
  heatmap images.
- NEW: Given 1–4 target metrics and a notion of "max is good" vs "min is good",
  propose two weight sets:
    - Candidate A: best observed run for those targets.
    - Candidate B: regression-guided synthetic configuration.
  Available via --mode analysis (with optional --targets) or --mode suggest.

Usage:
    python analyze_morl_wandb_runs.py path/to/wandb_export.csv \
        --mode analysis
        --targets ave_alive,morl/n1_proxy_mean_1000

    python analyze_morl_wandb_runs.py path/to/wandb_export.csv \
        --mode suggest \
        --targets ave_alive,morl/n1_proxy_mean_1000
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
from sklearn.neighbors import KernelDensity

# sklearn (regression + PCA)
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import r2_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# UMAP (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ---------------- CONFIG ----------------

# Default CSV if none provided on CLI
DEFAULT_CSV = "logs/wandb/wandb_export_2025-12-12T13_57_49.286+01_00.csv"

# Define which direction is "good" for each base metric (max or min)
# If a metric is missing from this dict, "max" is assumed.
METRIC_DIRECTION = {
    # Survival / longevity
    "ave_alive": "max",
    "morl/survival_mean_1000": "max",
    "morl/longevity_mean_1000": "max",

    # Fairness / equity
    "morl/fair_rho_mean_1000": "min",
    "morl/fair_curtail_mean_1000": "min",
    "morl/equity_curtail_mean_1000": "min",

    # Structural
    "morl/n1_proxy_mean_1000": "min",
    "morl/l2rpn_reward_mean_1000": "max",

    # Sustainability
    "morl/renewable_ratio_mean_1000": "max",
    "morl/co2_mean_1000": "min",

    # Cost / risk / simplicity (lower is better)
    "morl/econ_cost_mean_1000": "min",
    "morl/risk_mean_1000": "min",
    "morl/simplicity_mean_1000": "min",
}


# ---------------- CANONICAL ORDERING ----------------
# We keep plots and tables readable by grouping metrics/weights by MORL block.
# The order below is the canonical order used everywhere (base metrics, derived metrics, weights, etc.).
#
# Blocks in morl_objectives.py:
#   - Primary: survival
#   - Fairness: fair_rho, fair_curtail, equity_curtail
#   - Sustainability: renewable_ratio, co2
#   - Structural: risk, n1_proxy, econ_cost, simplicity, l2rpn_reward

# Base (true) metrics (stripped of optional "_norm")
BASE_METRIC_ORDER = [
    # Primary-ish / survival signals
    "ave_alive",
    "morl/survival_mean_1000",
    "morl/longevity_mean_1000",

    # Fairness block
    "morl/fair_rho_mean_1000",
    "morl/fair_curtail_mean_1000",
    "morl/equity_curtail_mean_1000",

    # Sustainability block
    "morl/renewable_ratio_mean_1000",
    "morl/co2_mean_1000",

    # Structural block
    "morl/risk_mean_1000",
    "morl/n1_proxy_mean_1000",
    "morl/econ_cost_mean_1000",
    "morl/simplicity_mean_1000",
    "morl/l2rpn_reward_mean_1000",
]

# Derived metrics (blocks + scalar reward) (stripped of optional "_norm")
DERIVED_METRIC_ORDER = [
    "morl/primary_block_mean_1000",
    "morl/fairness_block_mean_1000",
    "morl/sustain_block_mean_1000",
    "morl/structural_block_mean_1000",
    "morl/scalar_reward_mean_1000",
]

# Weights / hyper-parameters (stripped of optional "_norm")
WEIGHT_ORDER = [
    # Gate threshold + block scalers (most "conceptual" parameters)
    "morl/tau_primary",
    "morl/alpha_fair",
    "morl/alpha_sust",
    "morl/alpha_struct",

    # Fairness weights
    "morl/w_fair_rho",
    "morl/w_fair_curt",
    "morl/w_equity",

    # Sustainability weights
    "morl/w_ren",
    "morl/w_co2",

    # Structural weights
    "morl/w_risk",
    "morl/w_n1",
    "morl/w_econ",
    "morl/w_simplicity",
    "morl/w_l2rpn",


]
# Effective weights (alpha_* * w_*) produced by add_effective_weights()
EFF_WEIGHT_ORDER = [
    "eff_fair_rho",
    "eff_fair_curt",
    "eff_equity",
    "eff_ren",
    "eff_co2",
    "eff_risk",
    "eff_n1",
    "eff_econ",
    "eff_simplicity",
    "eff_l2rpn",
]

# Non-morl/ helper columns we often plot alongside weights
AUX_FEATURE_ORDER = [
    "alpha_total",
    "gate_active",
    "gate_margin",
]

def _strip_norm_suffix(col: str) -> str:
    # Helper used for ordering: drop plotting/analysis suffixes.
    if col.endswith("_util"):
        col = col[:-5]
    if col.endswith("_norm"):
        col = col[:-5]
    return col

def order_columns(cols, canonical_order):
    """
    Order `cols` so that:
      1) columns appearing in `canonical_order` come first (in that order),
      2) for each base name, the non-_norm version comes before _norm,
      3) interaction features like *_x_gate are placed right after their base,
      4) unknown columns are appended (stable alphabetical).
    """
    cols = list(dict.fromkeys(cols))  # de-dup, keep first occurrence
    base_index = {name: i for i, name in enumerate(canonical_order)}

    def key(c):
        base = _strip_norm_suffix(c)

        # Place "*_x_gate" right after its base
        is_gate_interaction = base.endswith("_x_gate")
        base_for_order = base[:-7] if is_gate_interaction else base

        idx = base_index.get(base_for_order, 10**9)

        # within the same base: raw then norm then interaction
        is_norm = c.endswith("_norm")
        within = 0
        if is_norm:
            within = 1
        if is_gate_interaction:
            within = 2

        return (idx, within, base, c)

    known = [c for c in cols if base_index.get(_strip_norm_suffix(c).replace("_x_gate",""), None) is not None]
    unknown = [c for c in cols if c not in known]
    # Sort all with key; unknown will drift to the end because idx=1e9
    return sorted(cols, key=key)

# Make pandas print full tables
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", None)

# Custom red-yellow-green colormap for table images / heatmaps
TABLE_CMAP = LinearSegmentedColormap.from_list(
    "ryg", ["red", "yellow", "green"]
)

# Mapping from effective weights to their parent raw weights
EFFECTIVE_PARENT = {
    "eff_fair_rho": "morl/w_fair_rho",
    "eff_fair_curt": "morl/w_fair_curt",
    "eff_equity": "morl/w_equity",
    "eff_ren": "morl/w_ren",
    "eff_co2": "morl/w_co2",
    "eff_risk": "morl/w_risk",
    "eff_n1": "morl/w_n1",
    "eff_econ": "morl/w_econ",
    "eff_simplicity": "morl/w_simplicity",
    "eff_l2rpn": "morl/w_l2rpn",
}

# NEW: mapping from effective weights back to their gating alpha
EFFECTIVE_ALPHA = {
    "eff_fair_rho": "morl/alpha_fair",
    "eff_fair_curt": "morl/alpha_fair",
    "eff_equity": "morl/alpha_fair",
    "eff_ren": "morl/alpha_sust",
    "eff_co2": "morl/alpha_sust",
    "eff_risk": "morl/alpha_struct",
    "eff_n1": "morl/alpha_struct",
    "eff_econ": "morl/alpha_struct",
    "eff_simplicity": "morl/alpha_struct",
    "eff_l2rpn": "morl/alpha_struct",
}

# ---------------- PLOT FUNCTIONS ----------------
def save_metrics_corr_heatmap_table_redesigned(
    corr_df,
    out_path,
    *,
    title="Metric–metric correlations",
    metric_direction=None,
    order=None,
    block_breaks=None,
    cmap="RdYlGn",
    vmin=-1.0,
    vmax=1.0,
    annotate=True,
    annot_fmt="{:.3f}",
    dpi=200,
):
    """
    Redesigned metric–metric correlation heatmap:
      - reorder rows/cols by `order` (base names, no suffixes needed)
      - robustly matches corr_df labels that may include *_util / *_norm suffixes
      - draws thick lines at `block_breaks`
      - colors tick labels by metric_direction (max/min/derived)
    """
    if metric_direction is None:
        metric_direction = {}

    def _base_name(s: str) -> str:
        # match your _strip_norm_suffix semantics: remove _util then _norm
        if s.endswith("_util"):
            s = s[:-5]
        if s.endswith("_norm"):
            s = s[:-5]
        return s

    # --- default order (your requested NEW order) ---
    if order is None:
        order = [
            "ave_alive",
            "morl/scalar_reward_mean_1000",

            "morl/primary_block_mean_1000",
            "morl/survival_mean_1000",
            "morl/longevity_mean_1000",

            "morl/fairness_block_mean_1000",
            "morl/fair_rho_mean_1000",
            "morl/fair_curtail_mean_1000",
            "morl/equity_curtail_mean_1000",

            "morl/sustain_block_mean_1000",
            "morl/renewable_ratio_mean_1000",
            "morl/co2_mean_1000",

            "morl/structural_block_mean_1000",
            "morl/risk_mean_1000",
            "morl/n1_proxy_mean_1000",
            "morl/econ_cost_mean_1000",
            "morl/simplicity_mean_1000",
            "morl/l2rpn_reward_mean_1000",
        ]

    # [2] global, [3] primary, [4] fairness, [3] sustain, [6] structural
    if block_breaks is None:
        block_breaks = [2, 5, 9, 12]

    # --- Build mapping base_name -> actual label in corr_df ---
    # corr_df is square (rows=cols), but we build from columns anyway
    col_map = {}
    for c in list(corr_df.columns):
        b = _base_name(str(c))
        # keep first occurrence
        if b not in col_map:
            col_map[b] = c

    # Resolve desired order into actual labels present in corr_df
    present_base = [b for b in order if b in col_map]
    missing_base = [b for b in order if b not in col_map]

    if len(present_base) < 2:
        # don’t crash the whole analysis: emit useful debug and return
        print("[WARN] metrics_corr_heatmap: not enough metrics found to plot after resolving names.")
        print(f"  corr_df columns (sample): {list(corr_df.columns)[:10]}")
        print(f"  missing from order: {missing_base}")
        return

    present_labels = [col_map[b] for b in present_base]

    # Reindex both rows and cols using actual labels
    corr = corr_df.loc[present_labels, present_labels].copy()

    # Plot
    n = len(present_labels)
    fig_w = max(7, 0.55 * n + 2)
    fig_h = max(6, 0.55 * n + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # ticks show base names (clean, consistent)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(present_base, rotation=90)
    ax.set_yticklabels(present_base)

    # label colors by objective direction
    def label_color(base_metric: str) -> str:
        direction = metric_direction.get(base_metric, None)
        if direction == "max":
            return "#1b9e77"
        if direction == "min":
            return "#d95f02"
        return "#444444"

    for lab in ax.get_xticklabels():
        lab.set_color(label_color(lab.get_text()))
    for lab in ax.get_yticklabels():
        lab.set_color(label_color(lab.get_text()))

    # annotate
    if annotate:
        data = corr.values.astype(float)
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, annot_fmt.format(val), ha="center", va="center", fontsize=7, color="black")

    # separators (still based on your intended block indices)
    for b in block_breaks:
        if 0 < b < n:
            ax.axhline(b - 0.5, color="black", linewidth=2.5)
            ax.axvline(b - 0.5, color="black", linewidth=2.5)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    if missing_base:
        print(f"[WARN] metrics_corr_heatmap: {len(missing_base)} ordered metrics missing and were skipped: {missing_base}")


def save_weights_vs_metrics_corr_heatmap_table_redesigned(
    corr_wm_df,
    out_path,
    *,
    title="Weight–metric correlations",
    metric_direction=None,
    rows_order=None,
    cols_order=None,
    row_block_breaks=None,
    col_block_breaks=None,
    cmap="RdYlGn",
    vmin=-1.0,
    vmax=1.0,
    annotate=True,
    annot_fmt="{:.3f}",
    dpi=200,
):
    """
    Redesigned weights-vs-metrics correlation heatmap:
      - corr_wm_df rows: weights/effective weights
      - corr_wm_df cols: metrics (may be *_util)
      - robustly matches cols_order base names to actual corr_wm_df columns
      - draws thick separators for row/col blocks
      - colors metric x-axis labels by objective direction
    """
    if metric_direction is None:
        metric_direction = {}

    def _base_name(s: str) -> str:
        if s.endswith("_util"):
            s = s[:-5]
        if s.endswith("_norm"):
            s = s[:-5]
        return s

    # --- defaults from your requested NEW order ---
    if rows_order is None:
        rows_order = [
            "morl/tau_primary",

            "morl/alpha_fair",
            "morl/w_fair_rho",
            "morl/w_fair_curt",
            "morl/w_equity",
            "eff_fair_rho",
            "eff_fair_curt",
            "eff_equity",

            "morl/alpha_sust",
            "morl/w_ren",
            "morl/w_co2",
            "eff_ren",
            "eff_co2",

            "morl/alpha_struct",
            "morl/w_risk",
            "morl/w_n1",
            "morl/w_econ",
            "morl/w_simplicity",
            "morl/w_l2rpn",
            "eff_risk",
            "eff_n1",
            "eff_econ",
            "eff_simplicity",
            "eff_l2rpn",
        ]

    if cols_order is None:
        cols_order = [
            "ave_alive",
            "morl/scalar_reward_mean_1000",

            "morl/primary_block_mean_1000",
            "morl/survival_mean_1000",
            "morl/longevity_mean_1000",

            "morl/fairness_block_mean_1000",
            "morl/fair_rho_mean_1000",
            "morl/fair_curtail_mean_1000",
            "morl/equity_curtail_mean_1000",

            "morl/sustain_block_mean_1000",
            "morl/renewable_ratio_mean_1000",
            "morl/co2_mean_1000",

            "morl/structural_block_mean_1000",
            "morl/risk_mean_1000",
            "morl/n1_proxy_mean_1000",
            "morl/econ_cost_mean_1000",
            "morl/simplicity_mean_1000",
            "morl/l2rpn_reward_mean_1000",
        ]

    # Rows: [1] tau, [7] fairness, [5] sust, [11] structural
    if row_block_breaks is None:
        row_block_breaks = [1, 8, 13]

    # Cols: [2] global, [3] primary, [4] fairness, [3] sustain, [6] structural
    if col_block_breaks is None:
        col_block_breaks = [2, 5, 9, 12]

    # --- rows (weights) must match exactly (they are not *_util) ---
    present_rows = [r for r in rows_order if r in corr_wm_df.index]
    missing_rows = [r for r in rows_order if r not in corr_wm_df.index]

    # --- cols (metrics) may be *_util, so resolve by base name ---
    col_map = {}
    for c in list(corr_wm_df.columns):
        b = _base_name(str(c))
        if b not in col_map:
            col_map[b] = c

    present_cols_base = [b for b in cols_order if b in col_map]
    missing_cols_base = [b for b in cols_order if b not in col_map]
    present_cols = [col_map[b] for b in present_cols_base]

    if len(present_rows) < 2 or len(present_cols) < 2:
        print("[WARN] weights_vs_metrics_heatmap: not enough rows/cols found to plot after resolving names.")
        print(f"  present_rows={len(present_rows)} present_cols={len(present_cols)}")
        return

    corr = corr_wm_df.loc[present_rows, present_cols].copy()

    # plot
    n_rows = len(present_rows)
    n_cols = len(present_cols)

    fig_w = max(8, 0.45 * n_cols + 4)
    fig_h = max(7, 0.35 * n_rows + 4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))

    # show base metric names on x-axis
    ax.set_xticklabels(present_cols_base, rotation=90)
    ax.set_yticklabels([_base_name(r) for r in present_rows])

    # x-label colors by direction
    def metric_label_color(base_metric: str) -> str:
        direction = metric_direction.get(base_metric, None)
        if direction == "max":
            return "#1b9e77"
        if direction == "min":
            return "#d95f02"
        return "#444444"

    for lab in ax.get_xticklabels():
        lab.set_color(metric_label_color(lab.get_text()))

    for lab in ax.get_yticklabels():
        lab.set_color("#111111")

    # annotate
    if annotate:
        data = corr.values.astype(float)
        for i in range(n_rows):
            for j in range(n_cols):
                val = data[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, annot_fmt.format(val), ha="center", va="center", fontsize=7, color="black")

    # separators (clip to plotted size)
    for b in row_block_breaks:
        if 0 < b < n_rows:
            ax.axhline(b - 0.5, color="black", linewidth=2.5)
    for b in col_block_breaks:
        if 0 < b < n_cols:
            ax.axvline(b - 0.5, color="black", linewidth=2.5)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson r")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    if missing_rows:
        print(f"[WARN] weights_vs_metrics_heatmap: {len(missing_rows)} ordered rows missing and were skipped: {missing_rows}")
    if missing_cols_base:
        print(f"[WARN] weights_vs_metrics_heatmap: {len(missing_cols_base)} ordered metrics missing and were skipped: {missing_cols_base}")


# ---------------- HELPER FUNCTIONS ----------------

def ensure_output_dir(dirname="analysis_outputs"):
    out_dir = Path(dirname)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_df_and_print(df: pd.DataFrame, path: Path, name: str | None = None):
    """Save a DataFrame to CSV and print the full table to stdout."""
    df.to_csv(path, index=False)
    label = name or path.name
    print(f"\n=== {label} (saved to {path}) ===")
    print(df)
    print("\n")

def save_df(df: pd.DataFrame, path: Path, name: str | None = None):
    """Save a DataFrame to CSV and print the full table to stdout."""
    df.to_csv(path, index=False)
    label = name or path.name
    print(f"\n=== {label} (saved to {path}) ===")

def save_table_heatmap(df: pd.DataFrame, path: Path, title: str):
    """
    Create an image of a 'table' with colored cells.

    Only numeric columns are visualized; index and column names are used
    as labels. Colors go from min (red) over mid (yellow) to max (green).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print(f"No numeric data in table for heatmap: {title}")
        return

    data = numeric_df.values.astype(float)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Auto figure size based on table size
    n_rows, n_cols = numeric_df.shape
    fig_w = max(6, 0.5 * n_cols + 2)
    fig_h = max(4, 0.4 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, cmap=TABLE_CMAP, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([_strip_norm_suffix(str(c)) for c in numeric_df.columns], rotation=90)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([_strip_norm_suffix(str(i)) for i in numeric_df.index])

    # Annotate cells with values
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                text = ""
            else:
                text = f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved table heatmap: {path}")


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    """Load CSV and filter to runs that have MORL weights logged."""
    df = pd.read_csv(csv_path)

    # Weight columns are morl/w_* and alpha/tau (non-metric, non-transformed)
    weight_cols = [
        c for c in df.columns
        if c.startswith("morl/")
        and "transformed" not in c
        and not c.endswith("_mean_1000")
    ]

    if not weight_cols:
        raise RuntimeError("No MORL weight columns (morl/*) found in CSV!")

    mask_morl = ~df[weight_cols].isna().all(axis=1)
    df_morl = df[mask_morl].copy().reset_index(drop=True)

    # Add a simple integer ID for each run (for plotting labels)
    df_morl["run_idx"] = np.arange(1, len(df_morl) + 1)

    print(f"Loaded {len(df)} runs, kept {len(df_morl)} MORL runs.")
    return df_morl


def select_metric_and_weight_cols(df: pd.DataFrame):
    """
    Identify:
      - base metrics (true objectives): all morl/*_mean_1000 except blocks &
        scalar_reward & transformed_* plus ave_alive
      - derived metrics: block_* and scalar_reward_* (still morl/*_mean_1000)
      - weights: all morl/* that are not metrics & not transformed
    """
    base_metric_cols = []
    derived_metric_cols = []

    for c in df.columns:
        if not c.startswith("morl/"):
            continue
        if "transformed" in c:
            continue
        if not c.endswith("_mean_1000"):
            continue

        # Block metrics and scalar reward are "derived"
        if "block" in c or "scalar_reward" in c:
            derived_metric_cols.append(c)
        else:
            base_metric_cols.append(c)

    # Add ave_alive as a base metric if present
    if "ave_alive" in df.columns:
        base_metric_cols.insert(0, "ave_alive")

    # Weight columns (alphas, taus, w_*, etc. – non-transformed, non-metric)
    weight_cols = [
        c for c in df.columns
        if c.startswith("morl/")
        and "transformed" not in c
        and not c.endswith("_mean_1000")
    ]

    # Apply canonical ordering for readability / interpretability
    base_metric_cols = order_columns(base_metric_cols, BASE_METRIC_ORDER)
    derived_metric_cols = order_columns(derived_metric_cols, DERIVED_METRIC_ORDER)
    weight_cols = order_columns(weight_cols, WEIGHT_ORDER)

    print(f"Identified {len(base_metric_cols)} base metrics, "
          f"{len(derived_metric_cols)} derived metrics, "
          f"{len(weight_cols)} weights.")
    return base_metric_cols, derived_metric_cols, weight_cols


def add_effective_weights(df: pd.DataFrame, weight_cols):
    """
    Add gating-aware "effective" weights: alpha_* * w_* combinations.

    Returns: df, eff_weight_cols (list of new column names)
    """
    eff_weight_cols = []

    def maybe_create_eff(name, alpha_col, w_col):
        nonlocal df, eff_weight_cols
        if alpha_col in df.columns and w_col in df.columns:
            eff_name = f"eff_{name}"
            df[eff_name] = df[alpha_col].astype(float) * df[w_col].astype(float)
            eff_weight_cols.append(eff_name)

    maybe_create_eff("fair_rho", "morl/alpha_fair", "morl/w_fair_rho")
    maybe_create_eff("fair_curt", "morl/alpha_fair", "morl/w_fair_curt")
    maybe_create_eff("equity", "morl/alpha_fair", "morl/w_equity")

    maybe_create_eff("ren", "morl/alpha_sust", "morl/w_ren")
    maybe_create_eff("co2", "morl/alpha_sust", "morl/w_co2")

    maybe_create_eff("risk", "morl/alpha_struct", "morl/w_risk")
    maybe_create_eff("n1", "morl/alpha_struct", "morl/w_n1")
    maybe_create_eff("econ", "morl/alpha_struct", "morl/w_econ")
    maybe_create_eff("simplicity", "morl/alpha_struct", "morl/w_simplicity")
    maybe_create_eff("l2rpn", "morl/alpha_struct", "morl/w_l2rpn")

    if eff_weight_cols:
        print(f"Created {len(eff_weight_cols)} effective weight columns.")
    else:
        print("No effective weight columns created (check alpha_*/w_* names).")

    return df, eff_weight_cols


def minmax_normalize(df: pd.DataFrame, metric_cols, out_dir: Path):
    """Add *_norm columns: min-max normalized metrics to [0, 1]."""
    norm_cols = []
    stats_rows = []

    for col in metric_cols:
        values = df[col].astype(float)
        vmin = values.min()
        vmax = values.max()

        if np.isclose(vmax, vmin):
            # constant metric: set to 0.5
            norm = np.full_like(values, 0.5, dtype=float)
        else:
            norm = (values - vmin) / (vmax - vmin)

        norm_name = f"{col}_norm"
        df[norm_name] = norm
        norm_cols.append(norm_name)
        stats_rows.append({"metric": col, "norm_col": norm_name,
                           "min": vmin, "max": vmax})

    stats_df = pd.DataFrame(stats_rows)
    stats_path = out_dir / "metric_minmax_stats.csv"
    save_df(stats_df, stats_path, name="metric_minmax_stats")

    # Also create a simple min/max heatmap (metric x [min,max])
    stats_heat_df = stats_df.set_index("metric")[["min", "max"]]
    save_table_heatmap(
        stats_heat_df,
        out_dir / "metric_minmax_stats_heatmap.png",
        "Metric min/max stats"
    )

    return df, norm_cols


def add_utility_aligned_columns(df: pd.DataFrame, cols, *, mode: str):
    """Create utility-aligned columns where 'higher = better' for all metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (not modified in-place).
    cols : list[str]
        Columns to transform. These can be raw metrics or *_norm columns.
    mode : {"raw", "norm"}
        - "raw": for direction=="min" -> multiply by -1
        - "norm": for direction=="min" -> (1 - x)
    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with additional *_util columns added.
    util_cols : list[str]
        Names of created utility-aligned columns in the same order as `cols`.
    """
    if mode not in ("raw", "norm"):
        raise ValueError(f"mode must be 'raw' or 'norm', got {mode!r}")

    df_out = df.copy()
    util_cols = []

    for col in cols:
        base = _strip_norm_suffix(col)  # remove _norm if present
        direction = METRIC_DIRECTION.get(base, "max")

        out_col = f"{col}_util"
        x = df_out[col].astype(float)

        if direction == "min":
            df_out[out_col] = (-x) if mode == "raw" else (1.0 - x)
        else:
            df_out[out_col] = x

        util_cols.append(out_col)

    return df_out, util_cols


def compute_pareto_ranks(df: pd.DataFrame, objective_cols_norm):
    """
    Compute Pareto ranks for the given normalized objectives (all maximized).

    Rank 1: non-dominated (Pareto front)
    Rank 2: front after removing rank-1 runs, etc.
    """
    data = df[objective_cols_norm].values.astype(float)
    n = data.shape[0]

    ranks = np.zeros(n, dtype=int)
    current_rank = 1
    remaining = np.arange(n)

    while remaining.size > 0:
        is_dominated = np.zeros(remaining.size, dtype=bool)
        for i_idx, i in enumerate(remaining):
            for j_idx, j in enumerate(remaining):
                if i == j:
                    continue
                # all >= and at least one >
                if np.all(data[j] >= data[i]) and np.any(data[j] > data[i]):
                    is_dominated[i_idx] = True
                    break

        front = remaining[~is_dominated]
        ranks[front] = current_rank
        remaining = remaining[is_dominated]
        current_rank += 1

    df["pareto_rank"] = ranks
    df["is_pareto_front"] = df["pareto_rank"] == 1

    # Distance to ideal (1,1,...,1) in normalized space
    ideal = np.ones(data.shape[1], dtype=float)
    dists = np.sqrt(((data - ideal) ** 2).sum(axis=1))
    df["ideal_distance"] = dists

    # Composite score: lower rank + closer to ideal
    max_rank = df["pareto_rank"].max()
    df["multiobj_score"] = (
        (max_rank + 1 - df["pareto_rank"]) / (max_rank + 1)
        + (data.shape[1] - dists) / data.shape[1]
    )

    return df


def _label_front_points_2d(ax, df, x_col, y_col):
    front = df[df["is_pareto_front"]]
    for _, row in front.iterrows():
        ax.text(row[x_col], row[y_col], str(int(row["run_idx"])),
                fontsize=7, ha="center", va="bottom")


def plot_pareto_2d(df, x_col, y_col, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = np.where(df["is_pareto_front"], "red", "gray")
    ax.scatter(df[x_col], df[y_col], c=colors, alpha=0.7, edgecolors="none")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Pareto Front in 2D: {x_col} vs {y_col}")

    # Highlight front
    front = df[df["is_pareto_front"]]
    if not front.empty:
        ax.scatter(front[x_col], front[y_col], color="red",
                   s=90, edgecolors="black")
        _label_front_points_2d(ax, df, x_col, y_col)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved 2D Pareto plot: {out_path}")


def plot_pareto_3d(df, x_col, y_col, z_col, out_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.where(df["is_pareto_front"], "red", "gray")
    ax.scatter(df[x_col], df[y_col], df[z_col], c=colors, alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title("3D Pareto Surface")

    # Label Pareto-front points
    front = df[df["is_pareto_front"]]
    for _, row in front.iterrows():
        ax.text(row[x_col], row[y_col], row[z_col],
                str(int(row["run_idx"])), fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved 3D Pareto plot: {out_path}")


def corr_heatmap(df, cols, out_path, title):
    """Simple correlation heatmap between selected columns (Pearson)."""
    corr = df[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(0.5 * len(cols) + 2,
                                    0.5 * len(cols) + 2))
    im = ax.imshow(corr.values, cmap=TABLE_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    labels = [_strip_norm_suffix(c) for c in cols]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation heatmap: {out_path}")
    return corr


def run_regressions(df, feature_cols, target_norm_cols, out_dir: Path,
                    prefix=""):
    """
    For each normalized target metric, fit Linear, Ridge, and Lasso regression
    vs all features (weights + effective weights). Save coefficient
    and summary tables (CSV + printed) and R²/coef heatmaps.
    """
    if not SKLEARN_AVAILABLE:
        print("sklearn not available, skipping regression models.")
        return None, None

    X = df[feature_cols].values.astype(float)

    summary_rows = []
    coef_rows = []

    for metric in target_norm_cols:
        y = df[metric].values.astype(float)
        if np.allclose(y, y[0]):
            # constant metric, regression not meaningful
            continue

        models = [
            ("linear", LinearRegression()),
            ("ridge", Ridge(alpha=1.0)),
            ("lasso", Lasso(alpha=0.01, max_iter=10000)),
        ]

        for model_name, model in models:
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            summary_rows.append({
                "metric": metric,
                "model": model_name,
                "r2": r2,
                "intercept": float(model.intercept_),
                "prefix": prefix,
            })
            for w_name, coef in zip(feature_cols, model.coef_):
                coef_rows.append({
                    "metric": metric,
                    "model": model_name,
                    "feature": w_name,
                    "coef": float(coef),
                    "prefix": prefix,
                })

    summary_df = pd.DataFrame(summary_rows)
    coef_df = pd.DataFrame(coef_rows)

    if not summary_df.empty:
        summary_name = (
            f"{prefix}regression_summary_by_metric.csv"
            if prefix else "regression_summary_by_metric.csv"
        )
        coef_name = (
            f"{prefix}regression_coefficients_full.csv"
            if prefix else "regression_coefficients_full.csv"
        )
        summary_path = out_dir / summary_name
        coef_path = out_dir / coef_name

        save_df(summary_df, summary_path, name=summary_name)
        save_df(coef_df, coef_path, name=coef_name)
        print(f"Saved regression summaries and coefficients ({prefix}).")

        # R² heatmap (metric x model)
        try:
            pivot_r2 = summary_df.pivot(
                index="metric", columns="model", values="r2"
            )
            save_table_heatmap(
                pivot_r2,
                out_dir / f"{prefix}regression_r2_heatmap.png",
                f"Regression R² ({prefix.strip('_') or 'base'})"
            )
        except Exception as e:
            print(f"Could not create R² heatmap for prefix {prefix}: {e}")

        # Ridge coefficient heatmap (metric x feature)
        try:
            ridge_df = coef_df[coef_df["model"] == "ridge"]
            if not ridge_df.empty:
                pivot_coef = ridge_df.pivot(
                    index="metric", columns="feature", values="coef"
                )
                save_table_heatmap(
                    pivot_coef,
                    out_dir / f"{prefix}regression_ridge_coef_heatmap.png",
                    f"Ridge coefficients ({prefix.strip('_') or 'base'})"
                )
        except Exception as e:
            print(f"Could not create ridge coef heatmap for prefix {prefix}: {e}")
    else:
        print(f"No regression results for prefix '{prefix}'.")

    return summary_df, coef_df


def run_pca_umap(df, metric_norm_cols_base, out_dir: Path):
    """
    Compute PCA and UMAP embeddings on normalized *base* metrics and save:
      - legacy plots (color=ave_alive_norm and color=pareto_rank)
      - NEW plots with richer semantics:
          color = dominant MORL block (primary/fairness/sustain/structural)
          size  = ave_alive_norm
          outline + labels for Pareto-front points
    """
    if not SKLEARN_AVAILABLE:
        print("sklearn not available, skipping PCA/UMAP.")
        return df

    import numpy as np
    import matplotlib.pyplot as plt

    # --- Helper: dominant block label per run ---
    def _compute_block_dominance(_df):
        block_cols = {
            "primary": "morl/primary_block_mean_1000",
            "fairness": "morl/fairness_block_mean_1000",
            "sustain": "morl/sustain_block_mean_1000",
            "structural": "morl/structural_block_mean_1000",
        }
        present = {k: v for k, v in block_cols.items() if v in _df.columns}
        if len(present) < 2:
            # Not enough info to compute dominance
            return None, []

        keys = list(present.keys())
        cols = [present[k] for k in keys]
        vals = _df[cols].to_numpy(dtype=float)

        # If NaNs exist, argmax can behave badly; replace NaN with -inf so it never wins
        vals = np.where(np.isfinite(vals), vals, -np.inf)

        idx = np.argmax(vals, axis=1)
        labels = np.array([keys[i] for i in idx], dtype=object)
        return labels, keys

    # --- Utility-align the normalized base metrics (so higher=better) ---
    df, metric_norm_cols_base_util = add_utility_aligned_columns(df, metric_norm_cols_base, mode="norm")
    metric_norm_cols_base_util = order_columns(metric_norm_cols_base_util, BASE_METRIC_ORDER)

    # Guard: remove missing cols (can happen if config changes)
    metric_norm_cols_base_util = [c for c in metric_norm_cols_base_util if c in df.columns]
    if len(metric_norm_cols_base_util) < 2:
        print("Not enough base metric columns present to run PCA/UMAP. Skipping.")
        return df

    X = df[metric_norm_cols_base_util].values.astype(float)

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    # --- UMAP ---
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)
        df["umap1"] = X_umap[:, 0]
        df["umap2"] = X_umap[:, 1]
    else:
        print("umap-learn not available, skipping UMAP embedding.")
        df["umap1"] = np.nan
        df["umap2"] = np.nan

    # --------- Legacy plots (keep for continuity) ----------
    # PCA color=ave_alive_norm
    if "ave_alive_norm" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(df["pca1"], df["pca2"], c=df["ave_alive_norm"], cmap="viridis", alpha=0.8)
        ax.set_xlabel("PCA1"); ax.set_ylabel("PCA2")
        ax.set_title("PCA of base metrics (utility-aligned) (color=ave_alive_norm)")
        front = df[df.get("is_pareto_front", False)]
        if "is_pareto_front" in df.columns:
            front = df[df["is_pareto_front"]]
            for _, row in front.iterrows():
                ax.text(row["pca1"], row["pca2"], str(int(row["run_idx"])),
                        fontsize=7, ha="center", va="bottom")
        fig.colorbar(sc, ax=ax, label="ave_alive_norm")
        fig.tight_layout()
        fig.savefig(out_dir / "pca_ave_alive_norm.png", dpi=150)
        plt.close(fig)

    # PCA color=pareto_rank
    if "pareto_rank" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(df["pca1"], df["pca2"], c=df["pareto_rank"], cmap="plasma", alpha=0.8)
        ax.set_xlabel("PCA1"); ax.set_ylabel("PCA2")
        ax.set_title("PCA of base metrics (utility-aligned) (color=pareto_rank)")
        if "is_pareto_front" in df.columns:
            front = df[df["is_pareto_front"]]
            for _, row in front.iterrows():
                ax.text(row["pca1"], row["pca2"], str(int(row["run_idx"])),
                        fontsize=7, ha="center", va="bottom")
        fig.colorbar(sc, ax=ax, label="pareto_rank")
        fig.tight_layout()
        fig.savefig(out_dir / "pca_pareto_rank.png", dpi=150)
        plt.close(fig)

    # UMAP legacy plots
    if UMAP_AVAILABLE:
        if "ave_alive_norm" in df.columns:
            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(df["umap1"], df["umap2"], c=df["ave_alive_norm"], cmap="viridis", alpha=0.8)
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
            ax.set_title("UMAP of base metrics (utility-aligned) (color=ave_alive_norm)")
            if "is_pareto_front" in df.columns:
                front = df[df["is_pareto_front"]]
                for _, row in front.iterrows():
                    ax.text(row["umap1"], row["umap2"], str(int(row["run_idx"])),
                            fontsize=7, ha="center", va="bottom")
            fig.colorbar(sc, ax=ax, label="ave_alive_norm")
            fig.tight_layout()
            fig.savefig(out_dir / "umap_ave_alive_norm.png", dpi=150)
            plt.close(fig)

        if "pareto_rank" in df.columns:
            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(df["umap1"], df["umap2"], c=df["pareto_rank"], cmap="plasma", alpha=0.8)
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
            ax.set_title("UMAP of base metrics (utility-aligned) (color=pareto_rank)")
            if "is_pareto_front" in df.columns:
                front = df[df["is_pareto_front"]]
                for _, row in front.iterrows():
                    ax.text(row["umap1"], row["umap2"], str(int(row["run_idx"])),
                            fontsize=7, ha="center", va="bottom")
            fig.colorbar(sc, ax=ax, label="pareto_rank")
            fig.tight_layout()
            fig.savefig(out_dir / "umap_pareto_rank.png", dpi=150)
            plt.close(fig)

    # --------- NEW: block-dominance plots (recommended for paper) ----------
    # Compute dominance labels (or skip if block cols missing)
    dom_labels, dom_keys = _compute_block_dominance(df)

    # Size scaling from ave_alive_norm (if available)
    if "ave_alive_norm" in df.columns:
        a = df["ave_alive_norm"].to_numpy(dtype=float)
        a = np.clip(a, 0.0, 1.0)
        sizes = 40.0 + 160.0 * a   # 40..200
    else:
        sizes = np.full(len(df), 80.0, dtype=float)

    # Pareto outline mask
    is_front = df["is_pareto_front"].to_numpy(dtype=bool) if "is_pareto_front" in df.columns else np.zeros(len(df), dtype=bool)

    # Choose distinct colors for categories (tab10 works well)
    cat_color = {
        "primary": "tab:blue",
        "fairness": "tab:green",
        "sustain": "tab:orange",
        "structural": "tab:red",
    }

    def _hex_to_rgb01(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0

    def _rgb01_to_hex(rgb):
        rgb = np.clip(rgb, 0.0, 1.0)
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    OKABE_ITO = {
        "primary": "#0072B2",
        "fairness": "#009E73",
        "sustain": "#E69F00",
        "structural": "#D55E00",
    }

    MARKERS = {
        "primary": "o",
        "fairness": "s",
        "sustain": "^",
        "structural": "D",
    }

    def _compute_block_shares(df):
        """Return (shares Nx4, keys) where shares rows sum to 1 (or 0 if invalid)."""
        block_cols = {
            "primary": "morl/primary_block_mean_1000",
            "fairness": "morl/fairness_block_mean_1000",
            "sustain": "morl/sustain_block_mean_1000",
            "structural": "morl/structural_block_mean_1000",
        }
        keys = [k for k, c in block_cols.items() if c in df.columns]
        cols = [block_cols[k] for k in keys]
        if len(keys) < 2:
            return None, None

        vals = df[cols].to_numpy(dtype=float)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        vals = np.clip(vals, 0.0, None)
        s = vals.sum(axis=1, keepdims=True)
        shares = np.divide(vals, s, out=np.zeros_like(vals), where=(s > 0))
        return shares, keys

    def _dominant_block_from_shares(shares, keys):
        idx = np.argmax(shares, axis=1)
        return np.array([keys[i] for i in idx], dtype=object)

    def _mix_facecolors(shares, keys):
        """Weighted RGB mix of block colors according to shares."""
        cols_rgb = np.stack([_hex_to_rgb01(OKABE_ITO[k]) for k in keys], axis=0)  # (K,3)
        mixed_rgb = shares @ cols_rgb  # (N,K)@(K,3) -> (N,3)
        return np.array([_rgb01_to_hex(rgb) for rgb in mixed_rgb], dtype=object)

    def _best_in_any_metric_mask(df, metric_cols):
        """
        True for runs that are top-1 in ANY of the given metric columns.
        metric_cols should already be utility-aligned so max is best.
        """
        if not metric_cols:
            return np.zeros(len(df), dtype=bool)
        mask = np.zeros(len(df), dtype=bool)
        for col in metric_cols:
            if col not in df.columns:
                continue
            x = df[col].to_numpy(dtype=float)
            if not np.any(np.isfinite(x)):
                continue
            m = np.nanmax(x)
            # allow ties within tiny tolerance
            mask |= np.isfinite(x) & (np.abs(x - m) <= 1e-12)
        return mask

    def _plot_block_map(xcol, ycol, out_name, main_title, subtitle, highlight_metric_cols=None):
        """
        Scatter map with:
          - marker shape = dominant block
          - facecolor = mixture of block shares
          - size = ave_alive_norm (bigger = survives longer)
          - shaded convex hull per dominant block (region shading)
          - highlights (shape-matching rings):
              * Pareto front: black ring
              * Best: union of TOP_K per metric: green ring
              * Worst: union of BOTTOM_K per metric: red ring
          - labels: stable 1..N order
        """



        # ---- Make indexing stable (prevents off-by-one / index drift) ----
        dfp = df.reset_index(drop=True).copy()
        n = len(dfp)
        point_ids = np.arange(n) + 1  # labels shown on plot

        # --- Block dominance (prefer *_norm_util, then *_norm, then *_util, then raw) ---
        block_base_cols = {
            "primary": "morl/primary_block_mean_1000",
            "fairness": "morl/fairness_block_mean_1000",
            "sustain": "morl/sustain_block_mean_1000",
            "structural": "morl/structural_block_mean_1000",
        }

        block_vals = {}
        for k, base in block_base_cols.items():
            cand = [f"{base}_norm_util", f"{base}_norm", f"{base}_util", base]
            col = next((c for c in cand if c in dfp.columns), None)
            if col is None:
                print(f"Skipping {out_name}: missing block col for '{k}' (tried {cand}).")
                return
            block_vals[k] = dfp[col].astype(float).values

        keys = list(block_vals.keys())
        mat = np.vstack([block_vals[k] for k in keys]).T  # (n,4)

        # defensive cleanup
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        mat = np.clip(mat, 0.0, None)
        row_sum = mat.sum(axis=1, keepdims=True)
        shares = np.where(row_sum > 0, mat / row_sum, 0.25)
        dom = np.array([keys[int(np.argmax(r))] for r in shares])

        # Color-blind friendly base palette (Okabe-Ito-ish RGB)
        base_colors = {
            "primary": (0 / 255, 114 / 255, 178 / 255),  # blue
            "fairness": (0 / 255, 158 / 255, 115 / 255),  # green
            "sustain": (230 / 255, 159 / 255, 0 / 255),  # orange
            "structural": (213 / 255, 94 / 255, 0 / 255),  # vermillion
        }

        # Mixture facecolors (weighted by shares)
        facecolors = np.zeros((n, 3), dtype=float)
        for j, k in enumerate(keys):
            facecolors += shares[:, [j]] * np.array(base_colors[k])[None, :]
        facecolors = np.clip(facecolors, 0.0, 1.0)

        # -----------------------------
        # NEW: size scaling by ave_alive_norm
        # -----------------------------
        if "ave_alive_norm" in dfp.columns:
            sraw = dfp["ave_alive_norm"].astype(float).values
            sraw = np.nan_to_num(sraw, nan=0.0, posinf=1.0, neginf=0.0)
            sraw = np.clip(sraw, 0.0, 1.0)
            # nice visual range; tune if you want
            sizes = 50.0 + 350.0 * sraw  # 90 .. 350
        else:
            sizes = np.full(n, 180.0, dtype=float)

        # --- Highlight masks (TOP_K / BOTTOM_K per metric, using *_norm_util if possible) ---
        TOP_K = 2
        BOTTOM_K = 2

        def _resolve_util_col(col: str):
            cand = [f"{col}_norm_util", f"{col}_util", f"{col}_norm", col]
            return next((c for c in cand if c in dfp.columns), None)

        best_any = np.zeros(n, dtype=bool)
        worst_any = np.zeros(n, dtype=bool)

        if highlight_metric_cols:
            for c in highlight_metric_cols:
                uc = _resolve_util_col(c)
                if uc is None:
                    continue
                x = dfp[uc].astype(float).values
                finite_idx = np.where(np.isfinite(x))[0]
                if finite_idx.size == 0:
                    continue

                top = finite_idx[np.argsort(x[finite_idx])[-TOP_K:]]
                best_any[top] = True

                bot = finite_idx[np.argsort(x[finite_idx])[:BOTTOM_K]]
                worst_any[bot] = True

        pareto_mask = (dfp["pareto_rank"].values == 1) if "pareto_rank" in dfp.columns else np.zeros(n, dtype=bool)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(11, 8.5))

        marker_map = {"primary": "o", "fairness": "s", "sustain": "^", "structural": "D"}
        legend_names = {
            "primary": "Primary",
            "fairness": "Fairness",
            "sustain": "Sustainability",
            "structural": "Structural",
        }
        # -----------------------------
        # NEW: density-based shaded regions per dominant block (KDE contours)
        # - produces complex shapes
        # - can yield multiple "islands" per block
        # - fairness won't "disappear": has a small-n fallback
        # -----------------------------


        def _shade_kde_regions(
                ax,
                xy,
                color_rgb,
                *,
                alpha=0.10,
                bandwidth=None,
                grid_size=220,
                quantiles=(0.80, 0.90, 0.96),
                zorder=1,
        ):
            """
            Shade high-density regions of a 2D point cloud using KDE + filled contours.

            - Complex shapes (not just ellipse/hull)
            - Multi-modal -> multiple shaded islands possible
            - For very small sample sizes, falls back to circles around points
            """
            xy = np.asarray(xy, dtype=float)
            n = xy.shape[0]

            # --- Small-n fallback (prevents "missing fairness") ---
            if n == 0:
                return
            if n < 4:
                # Draw soft blobs around points
                # radius scales with spread of the full plot (not perfect, but robust)
                r = 0.035 * max(
                    1e-9,
                    np.nanstd(dfp[xcol].values),
                    np.nanstd(dfp[ycol].values),
                )
                for p in xy:
                    ax.add_patch(
                        Circle(
                            (p[0], p[1]),
                            radius=r,
                            facecolor=color_rgb,
                            edgecolor="none",
                            alpha=alpha,
                            zorder=zorder,
                        )
                    )
                return

            # --- Bandwidth heuristic (robust across PCA vs UMAP scales) ---
            # If user didn't pass bandwidth, scale it with median pairwise distance.
            if bandwidth is None:
                # median distance to centroid -> stable, cheap
                c = np.mean(xy, axis=0, keepdims=True)
                d = np.sqrt(((xy - c) ** 2).sum(axis=1))
                md = float(np.median(d))
                bandwidth = max(1e-6, 0.35 * md)

            # --- Fit KDE ---
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
            kde.fit(xy)

            # --- Grid over local bbox (slightly padded) ---
            x = xy[:, 0]
            y = xy[:, 1]
            pad_x = 0.12 * (np.nanmax(x) - np.nanmin(x) + 1e-9)
            pad_y = 0.12 * (np.nanmax(y) - np.nanmin(y) + 1e-9)
            xmin, xmax = np.nanmin(x) - pad_x, np.nanmax(x) + pad_x
            ymin, ymax = np.nanmin(y) - pad_y, np.nanmax(y) + pad_y

            xs = np.linspace(xmin, xmax, grid_size)
            ys = np.linspace(ymin, ymax, grid_size)
            Xg, Yg = np.meshgrid(xs, ys)
            grid = np.column_stack([Xg.ravel(), Yg.ravel()])

            log_dens = kde.score_samples(grid)
            dens = np.exp(log_dens).reshape(grid_size, grid_size)

            if not np.any(np.isfinite(dens)) or float(np.nanmax(dens)) <= 0:
                return

            # --- Convert density to thresholds (quantile-based) ---
            flat = dens[np.isfinite(dens)].ravel()
            if flat.size < 50:
                return

            levels = []
            for q in quantiles:
                thr = float(np.quantile(flat, q))
                if thr > 0 and (not levels or thr > levels[-1] * 1.000001):
                    levels.append(thr)

            # If levels collapsed (rare), fall back to a single threshold
            if len(levels) == 0:
                levels = [float(np.quantile(flat, 0.90))]

            # Build contour levels from low->high and fill
            # We include the max so contourf fills properly
            levels = sorted(levels)
            levels = levels + [float(np.nanmax(flat))]

            ax.contourf(
                Xg, Yg, dens,
                levels=levels,
                colors=[color_rgb],
                alpha=alpha,
                antialiased=True,
                zorder=zorder,
            )

        # Draw KDE shading per block
        for k in keys:
            idx = (dom == k) & np.isfinite(dfp[xcol].values) & np.isfinite(dfp[ycol].values)
            if not np.any(idx):
                continue
            xy = np.column_stack([dfp.loc[idx, xcol].values, dfp.loc[idx, ycol].values])

            # Slightly different quantiles can help: UMAP often benefits from tighter (higher) thresholds.
            if xcol.startswith("umap"):
                qs = (0.70, 0.82, 0.92)
                a_shade = 0.10
            else:
                qs = (0.65, 0.80, 0.92)
                a_shade = 0.10


            _shade_kde_regions(
                ax,
                xy,
                color_rgb=base_colors[k],
                alpha=a_shade,
                bandwidth=None,  # auto
                grid_size=220,
                quantiles=qs,
                zorder=1,
            )

        _toggle_linewidth = 1.2

        # base points (by dominant block)
        for k in keys:
            idx = (dom == k)
            if not np.any(idx):
                continue
            ax.scatter(
                dfp.loc[idx, xcol], dfp.loc[idx, ycol],
                s=sizes[idx], marker=marker_map[k],
                c=facecolors[idx],
                edgecolors="0.15", linewidths=_toggle_linewidth, alpha=0.95, zorder=2
            )


        # labels (stable 1..N)
        for i in range(n):
            ax.text(dfp.loc[i, xcol], dfp.loc[i, ycol], str(point_ids[i]),
                    ha="center", va="center", fontsize=9, color="black", zorder=7)

        # --- titles and axis labels (pretty names) ---
        def _pretty_axis(s: str) -> str:
            s_low = (s or "").lower()
            if s_low.startswith("pca"):
                return f"PCA {s_low.replace('pca', '')}".strip()
            if s_low.startswith("umap"):
                return f"UMAP {s_low.replace('umap', '')}".strip()
            return (s or "").replace("_", " ").title()

        block_display = {
            "primary": "Primary",
            "fairness": "Fairness",
            "sustain": "Sustainability",
            "structural": "Structural",
        }

        # Title + subtitle as two separate text objects so we can control font sizes
        # and avoid overflow of long subtitles.
        fig.suptitle(main_title, fontsize=18, y=0.985)
        fig.text(
            0.5, 0.955, subtitle,
            ha="center", va="top",
            fontsize=12
        )

        ax.set_xlabel(_pretty_axis(xcol))
        ax.set_ylabel(_pretty_axis(ycol))
        ax.grid(True, alpha=0.25)

        from matplotlib.lines import Line2D

        dom_handles = [
            Line2D([0], [0], marker=marker_map[k], color="w",
                   label=block_display.get(k, k),
                   markerfacecolor=base_colors[k], markeredgecolor="0.15",
                   markersize=11, linewidth=0)
            for k in keys
        ]

        leg1 = ax.legend(handles=dom_handles, title="Dominant block (shape)", loc="upper left", framealpha=0.95)
        ax.add_artist(leg1)


        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / out_name, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_dir / out_name}")

    highlight_metric_cols = [
        "ave_alive",
        "morl/scalar_reward_mean_1000",
        "morl/survival_mean_1000",
        "morl/longevity_mean_1000",
        "morl/fair_rho_mean_1000",
        "morl/fair_curtail_mean_1000",
        "morl/equity_curtail_mean_1000",
        "morl/renewable_ratio_mean_1000",
        "morl/co2_mean_1000",
        "morl/risk_mean_1000",
        "morl/n1_proxy_mean_1000",
        "morl/econ_cost_mean_1000",
        "morl/simplicity_mean_1000",
        "morl/12rpn_reward_mean_1000",
        # optionally block means too:
        "morl/primary_block_mean_1000",
        "morl/fairness_block_mean_1000",
        "morl/sustain_block_mean_1000",
        "morl/structural_block_mean_1000",
    ]

    _plot_block_map(
        "pca1", "pca2", "pca_block_dominance.png",
        main_title="PCA: Dominant block",
        subtitle="Shape encodes dominant block, facecolor encodes the block mix, and size encodes Ave. Alive",
        highlight_metric_cols=highlight_metric_cols
    )

    if UMAP_AVAILABLE:
        _plot_block_map(
            "umap1", "umap2", "umap_block_dominance.png",
            main_title="UMAP: Dominant block",
            subtitle="Shape encodes dominant block, facecolor encodes the block mix, and border highlights best/Pareto runs",
            highlight_metric_cols=highlight_metric_cols
        )

    return df



def write_top_runs_summary(df: pd.DataFrame, out_dir: Path,
                           base_metric_cols, derived_metric_cols,
                           weight_cols, eff_weight_cols):
    """
    Create a human-readable summary of Pareto-front runs, with run_idx,
    name, key metrics and weights.
    """
    def get_run_name(row):
        for k in ["Name", "name", "run_name", "id", "run_id"]:
            if k in row.index:
                return str(row[k])
        return "<unknown>"

    front = df[df["is_pareto_front"]].copy()
    front = front.sort_values(["pareto_rank", "ideal_distance"])

    lines = []
    lines.append("=== Pareto-front runs summary ===\n\n")
    for _, row in front.iterrows():
        lines.append(f"Run_idx: {int(row['run_idx'])}\n")
        lines.append(f"Name   : {get_run_name(row)}\n")
        lines.append(f"Pareto rank  : {int(row['pareto_rank'])}\n")
        lines.append(f"Ideal dist   : {row['ideal_distance']:.3f}\n")
        lines.append(f"Multiobj score: {row['multiobj_score']:.3f}\n")

        # Key metrics
        lines.append("\nBase metrics:\n")
        for col in base_metric_cols:
            if col in row.index:
                lines.append(f"  {col:35s}: {row[col]:.4f}\n")

        lines.append("\nDerived metrics (blocks / scalar_reward):\n")
        for col in derived_metric_cols:
            if col in row.index:
                lines.append(f"  {col:35s}: {row[col]:.4f}\n")

        # Weights
        lines.append("\nWeights:\n")
        for w in weight_cols:
            lines.append(f"  {w:25s}: {row[w]:.3f}\n")

        if eff_weight_cols:
            lines.append("\nEffective weights (alpha * w):\n")
            for ew in eff_weight_cols:
                lines.append(f"  {ew:25s}: {row[ew]:.3f}\n")

        lines.append("\n" + "-" * 60 + "\n\n")

    out_path = out_dir / "top_runs_summary.txt"
    with open(out_path, "w") as f:
        f.write("".join(lines))
    print(f"Saved top-runs summary to {out_path}")


def suggest_weights_regression(coef_df: pd.DataFrame,
                               feature_cols,
                               metric_norm_cols_base,
                               out_dir: Path):
    """
    Build regression-based weight suggestions (generic, not metric-specific).

    We aggregate RIDGE coefficients for a set of base metrics:
      - ave_alive_norm (if present)
      - morl/renewable_ratio_mean_1000_norm (if present)
      - morl/n1_proxy_mean_1000_norm (if present)

    Only positive coefficients are used (we don't want to push features
    that hurt the metric).
    """
    if coef_df is None or coef_df.empty:
        print("No regression coefficients, skipping weight suggestions.")
        return

    target_metrics = []
    for base in ["ave_alive",
                 "morl/renewable_ratio_mean_1000",
                 "morl/n1_proxy_mean_1000"]:
        norm_name = f"{base}_norm"
        if norm_name in metric_norm_cols_base:
            target_metrics.append(norm_name)

    lines = []
    lines.append("=== Regression-based weight suggestions (generic) ===\n\n")
    lines.append("Target base metrics (normalized): " +
                 ", ".join(target_metrics) + "\n\n")

    if not target_metrics:
        lines.append("No target base metrics found in normalized columns.\n")
        out_path = out_dir / "weight_suggestions_regression.txt"
        with open(out_path, "w") as f:
            f.write("".join(lines))
        print(f"Saved empty regression-based weight suggestions to {out_path}")
        return

    agg = np.zeros(len(feature_cols), dtype=float)

    for metric in target_metrics:
        sub = coef_df[
            (coef_df["metric"] == metric) &
            (coef_df["model"] == "ridge")
        ]
        if sub.empty:
            continue
        # align features
        coef_vec = np.array(
            [sub[sub["feature"] == w]["coef"].values[0]
             if w in sub["feature"].values else 0.0
             for w in feature_cols],
            dtype=float,
        )
        # keep positive effects only
        coef_vec = np.maximum(0.0, coef_vec)
        if coef_vec.max() > 0:
            coef_vec = coef_vec / (coef_vec.max() + 1e-8)
        agg += coef_vec

    if agg.max() > 0:
        agg_norm = agg / agg.max()
    else:
        agg_norm = agg

    lines.append("Suggested relative feature strengths (0..1):\n")
    for w, val in zip(feature_cols, agg_norm):
        lines.append(f"  {w:30s}: {val:.3f}\n")

    lines.append("\nInterpretation:\n")
    lines.append(
        "- Values close to 1.0 are features (raw or effective weights)\n"
        "  that the regression models associate with higher scores on\n"
        "  the selected target base metrics.\n"
        "- Values near 0.0 are features that do not help, or that hurt\n"
        "  those metrics.\n"
    )

    out_path = out_dir / "weight_suggestions_regression.txt"
    with open(out_path, "w") as f:
        f.write("".join(lines))
    print(f"Saved regression-based weight suggestions to {out_path}")


def suggest_weights_for_targets(
    targets: list[str],
    reg_coef_base: pd.DataFrame,
    df_runs: pd.DataFrame,
    feature_cols: list[str],
    weight_cols: list[str],
    metric_norm_cols_base: list[str],
    out_dir: Path,
):
    """
    Given 1–4 target metrics (base metric names, e.g. 'ave_alive',
    'morl/n1_proxy_mean_1000'), use ridge regression coefficients and
    the METRIC_DIRECTION map to:
      - compute feature importance per metric and aggregate across metrics,
      - derive raw-weight importances,
      - propose two weight sets:
          * Candidate A: best observed run in df_runs for these metrics.
          * Candidate B: regression-guided synthetic config.

    Results are printed and saved to
      weight_suggestions_targets_<metric...>.txt
    """
    if reg_coef_base is None or reg_coef_base.empty:
        print("No base regression coefficients available; cannot suggest weights for targets.")
        return

    # Map base metric names -> normalized metric names
    target_norms = []
    directions = {}
    for m in targets:
        norm_name = f"{m}_norm"
        if norm_name not in metric_norm_cols_base:
            print(f"WARNING: normalized metric '{norm_name}' not found; skipping target '{m}'.")
            continue
        target_norms.append(norm_name)
        directions[norm_name] = METRIC_DIRECTION.get(m, "max")

    if not target_norms:
        print("No valid target metrics found for suggestions.")
        return

    # --- Feature importance per metric ---
    # feature_cols: raw + effective
    feature_importance = {f: [] for f in feature_cols}

    for norm_metric in target_norms:
        direction = directions[norm_metric]
        sub = reg_coef_base[
            (reg_coef_base["metric"] == norm_metric) &
            (reg_coef_base["model"] == "ridge")
        ]
        if sub.empty:
            print(f"WARNING: no ridge regression coefficients for '{norm_metric}', skipping.")
            continue

        coef_vec = np.array(
            [sub[sub["feature"] == f]["coef"].values[0]
             if f in sub["feature"].values else 0.0
             for f in feature_cols],
            dtype=float,
        )

        if direction == "max":
            desir = np.maximum(0.0, coef_vec)  # positive helps
        else:  # "min"
            desir = np.maximum(0.0, -coef_vec)  # negative helps (reduces metric)

        if desir.max() > 0:
            desir = desir / (desir.max() + 1e-8)

        for f_name, d_val in zip(feature_cols, desir):
            feature_importance[f_name].append(d_val)

    # Aggregate importance across metrics (mean)
    agg_feat_importance = {}
    for f_name, vals in feature_importance.items():
        if len(vals) == 0:
            agg = 0.0
        else:
            agg = float(np.mean(vals))
        agg_feat_importance[f_name] = agg

    feat_vals = np.array(list(agg_feat_importance.values()), dtype=float)
    if feat_vals.max() > 0:
        feat_vals_norm = feat_vals / feat_vals.max()
    else:
        feat_vals_norm = feat_vals

    agg_feat_importance_norm = {
        f: v for f, v in zip(agg_feat_importance.keys(), feat_vals_norm)
    }

    # --- Map feature importance back to raw weights ---
    # We want both:
    #   - direct influence of each raw weight (if used as a feature),
    #   - plus influence via effective (alpha * w) features.
    # For each eff feature, we push importance to BOTH the inner w_* and
    # the corresponding alpha_* so that we never suggest "big w_* with zero alpha_*".
    raw_importance = {w: 0.0 for w in weight_cols}

    # 1) direct contribution from raw features (alphas, taus, w_*)
    for w in weight_cols:
        if w in agg_feat_importance_norm:
            raw_importance[w] += agg_feat_importance_norm[w]

    # 2) contributions from effective features back to parent w_* AND alpha_*
    for eff_name, parent_w in EFFECTIVE_PARENT.items():
        if eff_name not in agg_feat_importance_norm:
            continue
        eff_imp = agg_feat_importance_norm[eff_name]

        # add to the inner block weight
        if parent_w in raw_importance:
            raw_importance[parent_w] += eff_imp

        # add to the corresponding alpha gate
        alpha_name = EFFECTIVE_ALPHA.get(eff_name)
        if alpha_name in raw_importance:
            raw_importance[alpha_name] += eff_imp

    raw_vals = np.array(list(raw_importance.values()), dtype=float)
    if raw_vals.max() > 0:
        raw_vals_norm = raw_vals / raw_vals.max()
    else:
        raw_vals_norm = raw_vals

    raw_importance_norm = {
        w: v for w, v in zip(raw_importance.keys(), raw_vals_norm)
    }

    # --- Candidate B: regression-guided synthetic config ---
    candidate_B = {}
    for w, imp in raw_importance_norm.items():
        if imp >= 0.66:
            candidate_B[w] = 1.0
        elif imp >= 0.33:
            candidate_B[w] = 0.5
        else:
            candidate_B[w] = 0.0

    # Normalize alphas if present
    alpha_keys = ["morl/alpha_fair", "morl/alpha_struct", "morl/alpha_sust"]
    alpha_vals = [candidate_B.get(a, 0.0) for a in alpha_keys]
    alpha_sum = sum(alpha_vals)
    if alpha_sum > 0:
        for a, val in zip(alpha_keys, alpha_vals):
            candidate_B[a] = float(val / alpha_sum)
    else:
        # Default split if everything is 0
        candidate_B["morl/alpha_fair"] = 0.4
        candidate_B["morl/alpha_struct"] = 0.4
        candidate_B["morl/alpha_sust"] = 0.2

    # Clamp tau_primary if present
    tau_key = "morl/tau_primary"
    if tau_key in candidate_B:
        tau_val = candidate_B[tau_key]
        tau_val = min(max(tau_val, 0.1), 0.9)
        candidate_B[tau_key] = tau_val

    # --- Candidate A: best existing run for the targets ---
    df_runs = df_runs.copy()
    scores = []
    for idx, row in df_runs.iterrows():
        tot = 0.0
        count = 0
        for m in targets:
            norm_name = f"{m}_norm"
            if norm_name not in df_runs.columns:
                continue
            val = float(row[norm_name])
            direction = METRIC_DIRECTION.get(m, "max")
            if direction == "max":
                s = val
            else:
                s = 1.0 - val
            tot += s
            count += 1
        if count == 0:
            scores.append(0.0)
        else:
            scores.append(tot / count)
    df_runs["multi_target_score"] = scores

    best_idx = int(df_runs["multi_target_score"].idxmax())
    best_run = df_runs.loc[best_idx]

    candidate_A = {w: float(best_run[w]) for w in weight_cols}

    # --- Output suggestions ---
    targets_slug = "_".join(t.replace("/", "-") for t in targets)
    out_path = out_dir / f"weight_suggestions_targets_{targets_slug}.txt"

    lines = []
    lines.append("=== Target-based weight suggestions ===\n\n")
    lines.append("Targets:\n")
    for m in targets:
        direction = METRIC_DIRECTION.get(m, "max")
        lines.append(f"  {m:35s}: direction={direction}\n")
    lines.append("\n")

    # Candidate A
    lines.append("Candidate A: best existing run (multi-target score)\n")
    lines.append(f"  run_idx      : {int(best_run['run_idx'])}\n")
    for k in ["Name", "name", "run_name", "id", "run_id"]:
        if k in best_run.index:
            lines.append(f"  run_name     : {best_run[k]}\n")
            break
    lines.append(f"  score        : {best_run['multi_target_score']:.4f}\n\n")
    lines.append("  Weights:\n")
    for w in weight_cols:
        lines.append(f"    {w:25s}: {candidate_A[w]:.3f}\n")
    lines.append("\n")

    # Candidate B
    lines.append("Candidate B: regression-guided synthetic configuration\n\n")
    lines.append("  Relative raw weight importance (0..1):\n")
    for w in weight_cols:
        lines.append(f"    {w:25s}: {raw_importance_norm[w]:.3f}\n")
    lines.append("\n  Suggested weights (0 / 0.5 / 1.0, normalized alphas):\n")
    for w in weight_cols:
        lines.append(f"    {w:25s}: {candidate_B[w]:.3f}\n")
    lines.append("\n")
    lines.append("Interpretation:\n")
    lines.append(
        "- Candidate A is guaranteed to be an actually tested configuration\n"
        "  that performed best on the chosen targets given your current runs.\n"
        "- Candidate B is an extrapolation based on the linear models; it is\n"
        "  a good candidate for a new experiment but not guaranteed to be\n"
        "  globally optimal.\n"
    )

    with open(out_path, "w") as f:
        f.write("".join(lines))

    print(f"\n=== Target-based suggestions for {targets} ===")
    print("".join(lines))
    print(f"Saved target-based suggestions to {out_path}")


# ---------------- MAIN MODES ----------------

def run_full_analysis(csv_path: Path, targets: list[str] | None = None):
    out_dir = ensure_output_dir()

    # Load and filter MORL runs
    df_morl = load_and_filter(csv_path)

    # Identify base metrics, derived metrics, and weights
    base_metric_cols, derived_metric_cols, weight_cols = \
        select_metric_and_weight_cols(df_morl)

    # Drop rows with NaN on any base metric
    df_morl = df_morl.dropna(subset=base_metric_cols).reset_index(drop=True)

    # Add effective weights (gating-aware)
    df_morl, eff_weight_cols = add_effective_weights(df_morl, weight_cols)

    # Normalize both base and derived metrics
    all_metric_cols = base_metric_cols + derived_metric_cols
    df_morl, all_metric_norm_cols = minmax_normalize(
        df_morl, all_metric_cols, out_dir
    )

    # Normalized base metrics only
    metric_norm_cols_base = [f"{c}_norm" for c in base_metric_cols]
    metric_norm_cols_base = [c for c in metric_norm_cols_base
                             if c in df_morl.columns]

    # Build Pareto front on normalized *base* metrics
    df_morl = compute_pareto_ranks(df_morl, metric_norm_cols_base)

    # Save filtered + ranked runs
    save_df_and_print(
        df_morl,
        out_dir / "morl_runs_filtered.csv",
        name="morl_runs_filtered"
    )
    save_df_and_print(
        df_morl[df_morl["is_pareto_front"]],
        out_dir / "morl_pareto_front.csv",
        name="morl_pareto_front"
    )

    # --- PLOTS: some 2D/3D Pareto views (raw metrics) ---
    if ("ave_alive" in df_morl.columns and
            "morl/structural_block_mean_1000" in df_morl.columns):
        plot_pareto_2d(
            df_morl,
            "ave_alive",
            "morl/structural_block_mean_1000",
            out_dir / "pareto_scatter_ave_alive_vs_structural.png",
        )

    if ("morl/primary_block_mean_1000" in df_morl.columns and
            "morl/structural_block_mean_1000" in df_morl.columns):
        plot_pareto_2d(
            df_morl,
            "morl/primary_block_mean_1000",
            "morl/structural_block_mean_1000",
            out_dir / "pareto_scatter_primary_vs_structural.png",
        )

    if all(c in df_morl.columns for c in [
        "ave_alive",
        "morl/fairness_block_mean_1000",
        "morl/sustain_block_mean_1000",
    ]):
        plot_pareto_3d(
            df_morl,
            "ave_alive",
            "morl/fairness_block_mean_1000",
            "morl/sustain_block_mean_1000",
            out_dir / "pareto_3d_ave_alive_fairness_sustain.png",
        )

    # --- CORRELATIONS (raw metrics) ---
    # Base + derived metrics together
    metric_cols_for_corr_raw = base_metric_cols + derived_metric_cols
    # Utility-align metrics so that "higher = better" for all objectives.
    # This prevents sign confusion in correlation/PCA/UMAP when some metrics are "min is good".
    df_corr, metric_cols_for_corr = add_utility_aligned_columns(
        df_morl,
        metric_cols_for_corr_raw,
        mode="raw",
    )

    # Keep the same readable order (grouped by block) after adding suffixes
    metric_cols_for_corr = order_columns(metric_cols_for_corr,
                                         BASE_METRIC_ORDER + DERIVED_METRIC_ORDER)

    corr_metrics = corr_heatmap(
        df_corr,
        metric_cols_for_corr,
        out_dir / "corr_heatmap_metrics.png",
        title="Correlation heatmap: base + derived metrics (utility-aligned, higher=better)",
    )
    corr_metrics_csv = corr_metrics.reset_index().rename(columns={"index": "metric"})
    save_df_and_print(
        corr_metrics_csv,
        out_dir / "correlations_metrics.csv",
        name="correlations_metrics"
    )
    ordered_weight_cols = order_columns(weight_cols, WEIGHT_ORDER)
    ordered_eff_cols = order_columns(eff_weight_cols, EFF_WEIGHT_ORDER)

    wm_cols = ordered_weight_cols + ordered_eff_cols + metric_cols_for_corr
    corr_all = df_corr[wm_cols].corr(method="pearson")
    corr_wm = corr_all.loc[ordered_weight_cols + ordered_eff_cols, metric_cols_for_corr]

    corr_heatmap(
        df_corr[wm_cols],
        wm_cols,
        out_dir / "corr_heatmap_weights_vs_metrics.png",
        title="Correlation heatmap: weights (+effective) & metrics (utility-aligned metrics)",
    )
    corr_wm_csv = corr_wm.reset_index().rename(columns={"index": "weight"})
    save_df_and_print(
        corr_wm_csv,
        out_dir / "correlations_weights_vs_metrics.csv",
        name="correlations_weights_vs_metrics"
    )
    """
    # Also table-style heatmaps for these correlation tables
    save_table_heatmap(
        corr_metrics,
        out_dir / "correlations_metrics_heatmap_table.png",
        "Metric–metric correlations"
    )
    save_table_heatmap(
        corr_wm,
        out_dir / "correlations_weights_vs_metrics_heatmap_table.png",
        "Weight–metric correlations"
    )
    """

    # --- Metric–metric redesigned table heatmap ---
    save_metrics_corr_heatmap_table_redesigned(
        corr_metrics,
        out_dir / "correlations_metrics_heatmap_table.png",
        title="Metric–metric correlations (grouped by block)",
        metric_direction=METRIC_DIRECTION,
    )

    # --- Weight–metric redesigned table heatmap ---
    save_weights_vs_metrics_corr_heatmap_table_redesigned(
        corr_wm,
        out_dir / "correlations_weights_vs_metrics_heatmap_table.png",
        title="Weight–metric correlations (grouped by block)",
        metric_direction=METRIC_DIRECTION,
    )

    # --- REGRESSIONS: base metric_norm ~ weights & effective weights ---
    feature_cols = (order_columns(weight_cols, WEIGHT_ORDER)
                   + order_columns(eff_weight_cols, EFF_WEIGHT_ORDER)
                   + [c for c in AUX_FEATURE_ORDER if c in df_morl.columns])
    reg_summary_base, reg_coef_base = run_regressions(
        df_morl,
        feature_cols=feature_cols,
        target_norm_cols=metric_norm_cols_base,
        out_dir=out_dir,
        prefix="base_",
    )

    # --- REGRESSIONS: derived metric_norm ~ weights & effective weights ---
    metric_norm_cols_derived = [f"{c}_norm" for c in derived_metric_cols
                                if f"{c}_norm" in df_morl.columns]
    if metric_norm_cols_derived:
        run_regressions(
            df_morl,
            feature_cols=feature_cols,
            target_norm_cols=metric_norm_cols_derived,
            out_dir=out_dir,
            prefix="derived_",
        )

    # --- PCA + UMAP on normalized base metrics ---
    df_morl = run_pca_umap(df_morl, metric_norm_cols_base, out_dir)

    # --- Top runs summary ---
    write_top_runs_summary(
        df_morl,
        out_dir,
        base_metric_cols=base_metric_cols,
        derived_metric_cols=derived_metric_cols,
        weight_cols=weight_cols,
        eff_weight_cols=eff_weight_cols,
    )

    # --- Generic regression-based weight suggestions ---
    if reg_coef_base is not None and not reg_coef_base.empty:
        suggest_weights_regression(
            reg_coef_base,
            feature_cols=feature_cols,
            metric_norm_cols_base=metric_norm_cols_base,
            out_dir=out_dir,
        )

    # --- Target-based suggestions (optional) ---
    if targets:
        print(f"\nGenerating target-based suggestions for: {targets}")
        suggest_weights_for_targets(
            targets=targets,
            reg_coef_base=reg_coef_base,
            df_runs=df_morl,
            feature_cols=feature_cols,
            weight_cols=weight_cols,
            metric_norm_cols_base=metric_norm_cols_base,
            out_dir=out_dir,
        )

    print("\nFull analysis complete. Check 'analysis_outputs' for CSVs, plots and suggestions.")


def run_suggest_only(csv_path: Path, targets: list[str]):
    """
    Lightweight mode: only compute the minimal pipeline needed to
    suggest weights for given targets:

      - load & filter runs
      - identify metrics & weights
      - add effective weights
      - normalize metrics
      - compute Pareto ranks (so candidate A score makes sense)
      - regressions on base metrics
      - target-based suggestions

    No correlation, PCA, extra plots, etc.
    """
    if not targets:
        print("ERROR: --mode suggest requires --targets to be specified.")
        sys.exit(1)

    out_dir = ensure_output_dir()

    df_morl = load_and_filter(csv_path)
    base_metric_cols, derived_metric_cols, weight_cols = \
        select_metric_and_weight_cols(df_morl)

    df_morl = df_morl.dropna(subset=base_metric_cols).reset_index(drop=True)
    df_morl, eff_weight_cols = add_effective_weights(df_morl, weight_cols)

    all_metric_cols = base_metric_cols + derived_metric_cols
    df_morl, all_metric_norm_cols = minmax_normalize(
        df_morl, all_metric_cols, out_dir
    )

    metric_norm_cols_base = [f"{c}_norm" for c in base_metric_cols]
    metric_norm_cols_base = [c for c in metric_norm_cols_base
                             if c in df_morl.columns]

    df_morl = compute_pareto_ranks(df_morl, metric_norm_cols_base)

    # Regressions: base metrics only
    feature_cols = (order_columns(weight_cols, WEIGHT_ORDER)
                   + order_columns(eff_weight_cols, EFF_WEIGHT_ORDER)
                   + [c for c in AUX_FEATURE_ORDER if c in df_morl.columns])
    reg_summary_base, reg_coef_base = run_regressions(
        df_morl,
        feature_cols=feature_cols,
        target_norm_cols=metric_norm_cols_base,
        out_dir=out_dir,
        prefix="base_",
    )

    suggest_weights_for_targets(
        targets=targets,
        reg_coef_base=reg_coef_base,
        df_runs=df_morl,
        feature_cols=feature_cols,
        weight_cols=weight_cols,
        metric_norm_cols_base=metric_norm_cols_base,
        out_dir=out_dir,
    )

    print("\nSuggest-only mode complete. Check 'analysis_outputs' for suggestions.")


def _build_feature_row_from_candidate(candidate: dict, feature_cols: list[str]) -> np.ndarray:
    """Build a single feature row in the exact feature_cols order."""
    row = np.zeros(len(feature_cols), dtype=float)
    # Pre-extract alphas for effective weights
    alpha_fair = float(candidate.get("morl/alpha_fair", 0.0))
    alpha_sust = float(candidate.get("morl/alpha_sust", 0.0))
    alpha_struct = float(candidate.get("morl/alpha_struct", 0.0))

    for i, f in enumerate(feature_cols):
        if f.startswith("eff_"):
            # effective weights: alpha_* * corresponding w_*
            if f == "eff_fair_rho":
                row[i] = alpha_fair * float(candidate.get("morl/w_fair_rho", 0.0))
            elif f == "eff_fair_curt":
                row[i] = alpha_fair * float(candidate.get("morl/w_fair_curt", 0.0))
            elif f == "eff_equity":
                row[i] = alpha_fair * float(candidate.get("morl/w_equity", 0.0))
            elif f == "eff_ren":
                row[i] = alpha_sust * float(candidate.get("morl/w_ren", 0.0))
            elif f == "eff_co2":
                row[i] = alpha_sust * float(candidate.get("morl/w_co2", 0.0))
            elif f == "eff_risk":
                row[i] = alpha_struct * float(candidate.get("morl/w_risk", 0.0))
            elif f == "eff_n1":
                row[i] = alpha_struct * float(candidate.get("morl/w_n1", 0.0))
            elif f == "eff_econ":
                row[i] = alpha_struct * float(candidate.get("morl/w_econ", 0.0))
            elif f == "eff_simplicity":
                row[i] = alpha_struct * float(candidate.get("morl/w_simplicity", 0.0))
            elif f == "eff_l2rpn":
                row[i] = alpha_struct * float(candidate.get("morl/w_l2rpn", 0.0))
            else:
                row[i] = 0.0
        else:
            row[i] = float(candidate.get(f, 0.0))
    return row


def _standardize_X(X: np.ndarray, eps: float = 1e-12):
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig = np.where(sig < eps, 1.0, sig)
    Xz = (X - mu) / sig
    return Xz, mu, sig


def _bootstrap_ridge_models(X: np.ndarray, Y: np.ndarray, n_boot: int = 50, alpha: float = 1.0, seed: int = 42):
    """Fit bootstrapped Ridge models per target. Returns list[list[(coef, intercept)]]."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    models = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        Yb = Y[idx]
        # fit one ridge per target to keep dependencies minimal
        per_target = []
        for j in range(Y.shape[1]):
            y = Yb[:, j]
            m = Ridge(alpha=alpha)
            m.fit(Xb, y)
            per_target.append((m.coef_.astype(float).copy(), float(m.intercept_)))
        models.append(per_target)
    return models


def _predict_bootstrap(models, x_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and std across bootstraps for each target."""
    preds = []
    for per_target in models:
        yhat = []
        for (coef, intercept) in per_target:
            yhat.append(float(np.dot(coef, x_row) + intercept))
        preds.append(yhat)
    preds = np.asarray(preds, dtype=float)  # (B, T)
    return preds.mean(axis=0), preds.std(axis=0)


def _generate_candidate_weights(weight_cols: list[str], n_candidates: int, seed: int = 42) -> list[dict]:
    """
    Generate candidate weight configurations respecting:
      - alpha simplex on {alpha_fair, alpha_sust, alpha_struct}
      - tau_primary in [0.1, 0.9]
      - if alpha of a block is near 0, inner weights of that block set to 0
    """
    rng = np.random.default_rng(seed)
    candidates = []

    # inner weight groups
    fair_ws = ["morl/w_fair_rho", "morl/w_fair_curt", "morl/w_equity"]
    sust_ws = ["morl/w_ren", "morl/w_co2"]
    struct_ws = ["morl/w_risk", "morl/w_n1", "morl/w_econ", "morl/w_simplicity", "morl/w_l2rpn"]

    ALPHA_MIN = 0.05

    for _ in range(n_candidates):
        cand = {w: 0.0 for w in weight_cols}

        # alphas (Dirichlet)
        if all(a in weight_cols for a in ["morl/alpha_fair", "morl/alpha_sust", "morl/alpha_struct"]):
            a = rng.dirichlet([1.0, 1.0, 1.0])
            cand["morl/alpha_fair"] = float(a[0])
            cand["morl/alpha_sust"] = float(a[1])
            cand["morl/alpha_struct"] = float(a[2])

        # tau
        if "morl/tau_primary" in weight_cols:
            cand["morl/tau_primary"] = float(rng.uniform(0.1, 0.9))

        # sample inner weights
        def sample_block(ws):
            for w in ws:
                if w in weight_cols:
                    cand[w] = float(rng.uniform(0.0, 1.0))

        sample_block(fair_ws)
        sample_block(sust_ws)
        sample_block(struct_ws)

        # enforce alpha-gating of inner weights
        if "morl/alpha_fair" in cand and cand["morl/alpha_fair"] < ALPHA_MIN:
            for w in fair_ws:
                if w in cand:
                    cand[w] = 0.0
        if "morl/alpha_sust" in cand and cand["morl/alpha_sust"] < ALPHA_MIN:
            for w in sust_ws:
                if w in cand:
                    cand[w] = 0.0
        if "morl/alpha_struct" in cand and cand["morl/alpha_struct"] < ALPHA_MIN:
            for w in struct_ws:
                if w in cand:
                    cand[w] = 0.0

        candidates.append(cand)

    # Add a few explicit interaction probes (pairs high together) if possible
    probes = [
        ("morl/w_ren", "morl/w_co2"),
        ("morl/w_risk", "morl/w_n1"),
        ("morl/w_econ", "morl/w_simplicity"),
        ("morl/w_fair_rho", "morl/w_equity"),
    ]
    for (x, y) in probes:
        if x in weight_cols and y in weight_cols:
            cand = {w: 0.0 for w in weight_cols}
            # balanced alphas
            for a in ["morl/alpha_fair", "morl/alpha_sust", "morl/alpha_struct"]:
                if a in weight_cols:
                    cand[a] = 1/3
            if "morl/tau_primary" in weight_cols:
                cand["morl/tau_primary"] = 0.5
            cand[x] = 1.0
            cand[y] = 1.0
            candidates.append(cand)

    return candidates


def run_suggest_explore_only(csv_path: Path, n_suggestions: int = 4,
                             n_candidates: int = 2000, n_boot: int = 50,
                             seed: int = 42):
    """
    Exploration-focused suggestions: propose configurations that reduce model uncertainty.

    Pipeline (minimal):
      - load & filter runs
      - identify metrics & weights
      - add effective weights
      - normalize derived metrics
      - train bootstrapped Ridge ensemble to predict derived metrics from weights
      - sample candidate weight vectors and score by predictive uncertainty + novelty
      - output top `n_suggestions` (default 4) configs
    """
    if not SKLEARN_AVAILABLE:
        print("sklearn not available, cannot run suggest_explore.")
        sys.exit(1)

    out_dir = ensure_output_dir()

    df_morl = load_and_filter(csv_path)
    base_metric_cols, derived_metric_cols, weight_cols = select_metric_and_weight_cols(df_morl)

    # We need derived metrics present for targets; drop NaNs
    df_morl = df_morl.dropna(subset=derived_metric_cols).reset_index(drop=True)

    df_morl, eff_weight_cols = add_effective_weights(df_morl, weight_cols)

    # Normalize derived metrics only (these are the most meaningful "structured" targets)
    df_morl, _ = minmax_normalize(df_morl, derived_metric_cols, out_dir)
    target_norm_cols = [f"{c}_norm" for c in derived_metric_cols if f"{c}_norm" in df_morl.columns]

    # Utility-align normalized targets (usually no-op for derived metrics, but keeps semantics consistent)
    df_morl, target_norm_cols_util = add_utility_aligned_columns(df_morl, target_norm_cols, mode="norm")
    target_norm_cols_util = order_columns(target_norm_cols_util, DERIVED_METRIC_ORDER)

    # Features
    feature_cols = (order_columns(weight_cols, WEIGHT_ORDER)
                    + order_columns(eff_weight_cols, EFF_WEIGHT_ORDER)
                    + [c for c in AUX_FEATURE_ORDER if c in df_morl.columns])

    X = df_morl[feature_cols].values.astype(float)
    Xz, mu, sig = _standardize_X(X)
    Y = df_morl[target_norm_cols_util].values.astype(float)

    # Fit bootstrapped ridge models (per target)
    models = _bootstrap_ridge_models(Xz, Y, n_boot=n_boot, alpha=1.0, seed=seed)

    # Candidate generation
    candidates = _generate_candidate_weights(weight_cols, n_candidates=n_candidates, seed=seed)

    # Evaluate candidates
    existing_X = Xz
    # novelty: distance to nearest existing run (in standardized feature space)
    def novelty_score(xz):
        d = np.sqrt(((existing_X - xz) ** 2).sum(axis=1))
        return float(np.min(d))

    scored = []
    for cand in candidates:
        x_row = _build_feature_row_from_candidate(cand, feature_cols)
        xz = (x_row - mu) / sig
        mean_pred, std_pred = _predict_bootstrap(models, xz)

        # uncertainty: mean std across targets
        unc = float(np.mean(std_pred))
        nov = novelty_score(xz)

        scored.append((unc, nov, cand, mean_pred, std_pred))

    # Normalize novelty to 0..1 for scoring
    novs = np.array([s[1] for s in scored], dtype=float)
    if novs.max() > 1e-12:
        nov_norm = (novs - novs.min()) / (novs.max() - novs.min() + 1e-12)
    else:
        nov_norm = np.zeros_like(novs)

    final = []
    for i, (unc, nov, cand, mean_pred, std_pred) in enumerate(scored):
        # Score: prioritize uncertainty, but avoid duplicates via novelty
        score = unc * (0.5 + 0.5 * float(nov_norm[i]))
        final.append((score, unc, nov, cand, mean_pred, std_pred))

    final.sort(key=lambda x: x[0], reverse=True)
    top = final[:n_suggestions]

    # Save results
    rows = []
    for rank, (score, unc, nov, cand, mean_pred, std_pred) in enumerate(top, start=1):
        row = {"rank": rank, "explore_score": score, "uncertainty_mean_std": unc, "novelty_min_dist": nov}
        for w in weight_cols:
            row[w] = float(cand.get(w, 0.0))
        for t_name, mp, sp in zip(target_norm_cols_util, mean_pred, std_pred):
            row[f"pred_mean_{t_name}"] = float(mp)
            row[f"pred_std_{t_name}"] = float(sp)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    out_csv = out_dir / "weight_suggestions_explore.csv"
    save_df_and_print(df_out, out_csv, name="weight_suggestions_explore")

    # also human-readable txt
    out_txt = out_dir / "weight_suggestions_explore.txt"
    lines = []
    lines.append("=== Exploration-focused weight suggestions (suggest_explore) ===\n\n")
    lines.append(f"Candidates sampled: {len(candidates)}\n")
    lines.append(f"Bootstrap models   : {n_boot}\n")
    lines.append(f"Suggested configs  : {n_suggestions}\n\n")
    lines.append("Targets (utility-aligned):\n")
    for t in target_norm_cols_util:
        lines.append(f"  - {t}\n")
    lines.append("\n")

    for rank, (score, unc, nov, cand, mean_pred, std_pred) in enumerate(top, start=1):
        lines.append(f"[{rank}] explore_score={score:.4f}  uncertainty={unc:.4f}  novelty={nov:.4f}\n")
        lines.append("  Weights:\n")
        for w in weight_cols:
            lines.append(f"    {w:25s}: {float(cand.get(w, 0.0)):.3f}\n")
        lines.append("  Predicted targets (mean ± std):\n")
        for t_name, mp, sp in zip(target_norm_cols_util, mean_pred, std_pred):
            lines.append(f"    {_strip_norm_suffix(t_name):35s}: {mp:.3f} ± {sp:.3f}\n")
        lines.append("\n")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Saved exploration suggestions to {out_txt}")

    print("\nSuggest-explore mode complete. Check 'analysis_outputs' for explore suggestions.\n")


# ---------------- ENTRYPOINT ----------------

def main(csv_path: str | Path = DEFAULT_CSV,
         mode: str = "analysis",
         targets: list[str] | None = None):
    """
    Simple entrypoint without argparse.

    Parameters
    ----------
    csv_path : str or Path
        Path to the W&B export CSV.
    mode : {"analysis", "suggest", "suggest_explore"}
        - "analysis": full pipeline (Pareto, correlations, regressions, PCA/UMAP, plots)
        - "suggest": lightweight mode that only computes the minimum needed to
                     propose weights for the given targets
        - "suggest_explore": propose 4 exploratory configurations that maximize
                            model uncertainty reduction (active learning)
    targets : list[str] or None
        List of base metric names to optimise, e.g.
        ["ave_alive", "morl/n1_proxy_mean_1000"].
        Can be None or empty if you just want the generic analysis.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    # Normalise targets argument
    if targets is None:
        targets = []
    else:
        # Strip spaces, drop empties
        targets = [t.strip() for t in targets if t.strip()]

    if len(targets) > 4:
        print("WARNING: more than 4 targets given; only the first 4 will be used.")
        targets = targets[:4]

    if mode == "analysis":
        run_full_analysis(csv_path, targets=targets)
    elif mode == "suggest":
        run_suggest_only(csv_path, targets=targets)
    elif mode == "suggest_explore":
        run_suggest_explore_only(csv_path, n_suggestions=4)
    else:
        print(f"Unknown mode '{mode}'. Use 'analysis' or 'suggest'.")
        sys.exit(1)


if __name__ == "__main__":
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # EDIT THESE THREE LINES TO CONTROL WHAT THE SCRIPT DOES
    # Path to your W&B export CSV
    CSV_PATH = "logs/wandb/wandb_export_2025-12-15T23_46_51.468+01_00.csv"  # or "logs/wandb/my_export.csv"

    # Mode: "analysis" for full pipeline, "suggest" for suggestion-only
    MODE = "analysis"       # or "suggest", "analysis", "suggest_explore"


    # Targets: up to 4 base metrics you care about
    # (leave empty list [] if you don't want target-based suggestions)
    TARGETS = [
         "ave_alive"
    ]
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    main(CSV_PATH, MODE, TARGETS)