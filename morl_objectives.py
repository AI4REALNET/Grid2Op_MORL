from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd


# ===================================================================
# PARAMS
# ===================================================================
@dataclass
class MorlParams:
    """Configuration + metadata for MORL metrics.

    Most fields are filled automatically by `build_morl_params_from_dataset`.
    """
    # Scaling constants
    econ_scale: float = 1.0
    co2_scale: float = 1.0
    fairness_scale: float = 1.0
    risk_scale: float = 1.0
    simplicity_scale: float = 1.0
    n1_scale: float = 1.0

    # Economic parameters
    redispatch_cost_per_mw: float = 1.0
    curtailment_cost_per_mw: float = 1.0

    # Metadata (filled from dataset files if available)
    gen_emission: Optional[Dict[int, float]] = None          # generator_id -> tCO2/MWh (relative)
    renewable_gen_ids: Optional[set] = None                  # set of generator_ids considered renewable
    line_to_zone: Optional[np.ndarray] = None                # shape (n_line,), zone id per line
    gen_to_zone: Optional[np.ndarray] = None                 # shape (n_gen,), zone id per generator
    zone_population: Optional[Dict[int, float]] = None       # zone_id -> "population" proxy (e.g. MW of load)
    critical_lines: Optional[np.ndarray] = None              # indices of lines used in N-1 proxy


# ===================================================================
# HELPER
# ===================================================================
def scalarize_with_preferences(metrics, w, tau_primary=0.0):
    """
    Preference-conditioned scalarization.

    w: array-like of shape (4,) with
       w[0] -> primary robustness (longevity) weight
       w[1] -> fairness weight
       w[2] -> sustain weight
       w[3] -> structural weight
    """
    # 1) Reuse the same block construction as in build_gated_scalar_reward
    #    (you can literally copy that code, but without fixed alpha_*)
    primary_block = metrics.get(
        "transformed_longevity",
        metrics.get("transformed_survival", 0.0),
    )

    fairness_block = (
        -metrics.get("transformed_fair_rho", 0.0)
        -metrics.get("transformed_fair_curtail", 0.0)
        -metrics.get("transformed_equity_curtail", 0.0)
    )

    sustain_block = (
        +metrics.get("transformed_renewable_ratio", 0.0)
        -metrics.get("transformed_co2", 0.0)
    )

    structural_block = (
        -metrics.get("transformed_risk", 0.0)
        +metrics.get("transformed_n1_proxy", 0.0)
        -metrics.get("transformed_econ_cost", 0.0)
        +metrics.get("transformed_simplicity", 0.0)
        +metrics.get("transformed_l2rpn_reward", 0.0)
    )

    w_surv, w_fair, w_sust, w_struct = [float(x) for x in w]

    if primary_block < tau_primary:
        scalar_reward = w_surv * primary_block
    else:
        scalar_reward = (
            w_surv * primary_block
            + w_fair * fairness_block
            + w_sust * sustain_block
            + w_struct * structural_block
        )

    return {
        "scalar_reward": scalar_reward,
        "primary_block": primary_block,
        "fairness_block": fairness_block,
        "sustain_block": sustain_block,
        "structural_block": structural_block,
    }

def _safe_array(x, default=None):
    if x is None:
        return default
    return np.asarray(x)

def _read_table_from_grid_json(grid_data, name: str) -> pd.DataFrame:
    """Read a pandas-encoded table from grid.json without FutureWarnings."""
    spec = grid_data["_object"][name]
    # spec["_object"] is a JSON string -> wrap in StringIO as recommended
    return pd.read_json(StringIO(spec["_object"]), orient=spec["orient"])

def _build_bus_regions_grid_layout(
    layout: dict,
    bus_df: pd.DataFrame,
    n_cols: int = 4,
    n_rows: int = 3,
) -> np.ndarray:
    """
    Assign each substation (bus) to a region using a fixed 3x4 grid over
    the layout coordinates.

    - Columns (x direction): A, B, C, D  -> indices 0..3
    - Rows    (y direction): 1, 2, 3     -> indices 0..2
    - Region id = row * n_cols + col -> 0..11

    This is a deterministic "grid overlay" like in your screenshot and
    does not depend on any dataset-internal zone labels.
    """
    xs = []
    ys = []

    for name in bus_df["name"]:
        coord = layout.get(str(name))
        if coord is None:
            # If something is missing (shouldn't happen), put it at (0,0)
            xs.append(0.0)
            ys.append(0.0)
        else:
            xs.append(float(coord[0]))
            ys.append(float(coord[1]))

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    # Define uniform bins over x and y
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    if xmin == xmax:
        # Degenerate case: all same x -> single column
        col_idx = np.zeros_like(xs, dtype=int)
    else:
        col_edges = np.linspace(xmin, xmax, n_cols + 1)
        # digitize into 0..n_cols-1
        col_idx = np.digitize(xs, col_edges[1:-1], right=True).astype(int)

    if ymin == ymax:
        row_idx = np.zeros_like(ys, dtype=int)
    else:
        row_edges = np.linspace(ymin, ymax, n_rows + 1)
        row_idx = np.digitize(ys, row_edges[1:-1], right=True).astype(int)

    # Region = row * n_cols + col  (0..n_rows*n_cols-1)
    regions = row_idx * n_cols + col_idx
    return regions.astype(int)

# ===================================================================
# DATASET-DRIVEN PARAM BUILDER
# ===================================================================
def build_morl_params_from_dataset(
    dataset_path: str,
    econ_scale: float = 1.0,
    co2_scale: float = 1.0,
    fairness_scale: float = 1.0,
    risk_scale: float = 1.0,
    simplicity_scale: float = 1.0,
    n1_scale: float = 1.0,
    redispatch_cost_per_mw: float = 1.0,
    curtailment_cost_per_mw: float = 1.0,
) -> MorlParams:
    """
    Construct a MorlParams object by reading the standard Grid2Op dataset
    files for NeurIPS Track 1 style grids.

    We *ignore* any dataset-provided zones and instead:

    - read grid_layout.json,
    - overlay a fixed 3x4 grid (A-D x 1-3),
    - assign each substation (bus) to one of 12 regions,
    - map lines and generators to their substation regions.
    """

    ds_path = Path(dataset_path)

    # --- Load grid.json (pandas-encoded) ---
    with open(ds_path / "grid.json", "r") as f:
        grid_data = json.load(f)

    bus_df = _read_table_from_grid_json(grid_data, "bus")
    gen_df = _read_table_from_grid_json(grid_data, "gen")
    line_df = _read_table_from_grid_json(grid_data, "line")
    load_df = _read_table_from_grid_json(grid_data, "load")

    # ===============================================================
    # 1) BUS -> REGION mapping: fixed 3x4 grid via grid_layout.json
    # ===============================================================
    layout_path = ds_path / "grid_layout.json"
    if layout_path.exists():
        with open(layout_path, "r") as f:
            layout = json.load(f)
        bus_to_region = _build_bus_regions_grid_layout(layout, bus_df)
    else:
        # Fallback: single region if layout is missing
        bus_to_region = np.zeros(len(bus_df), dtype=int)

    # ===============================================================
    # 2) LINES & GENERATORS -> REGION
    # ===============================================================
    from_bus = line_df["from_bus"].to_numpy(dtype=int)
    to_bus = line_df["to_bus"].to_numpy(dtype=int)

    line_to_zone = bus_to_region[from_bus]
    bad = (line_to_zone < 0) | (line_to_zone >= bus_to_region.size)
    if np.any(bad):
        line_to_zone[bad] = bus_to_region[to_bus[bad]]
    line_to_zone[line_to_zone < 0] = 0

    gen_bus = gen_df["bus"].to_numpy(dtype=int)
    gen_to_zone = bus_to_region[gen_bus]

    # ===============================================================
    # 3) ZONE "POPULATION" from loads (per-region load)
    # ===============================================================
    load_bus = load_df["bus"].to_numpy(dtype=int)
    load_zone_ids = bus_to_region[load_bus]
    load_p = np.clip(load_df["p_mw"].to_numpy(dtype=float), 0.0, None)

    n_zones = int(np.max(load_zone_ids) + 1) if load_zone_ids.size > 0 else 1
    zone_population: Dict[int, float] = {}
    for z in range(n_zones):
        zone_population[z] = float(load_p[load_zone_ids == z].sum())

    # Avoid zeros to prevent division issues
    for z in zone_population:
        if zone_population[z] <= 0.0:
            zone_population[z] = 1.0

    # ===============================================================
    # 4) Generator characteristics from prods_charac.csv
    # ===============================================================
    prods_df = pd.read_csv(ds_path / "prods_charac.csv")

    base_emissions = {
        "thermal": 1.0,
        "gas": 0.7,
        "coal": 1.2,
        "lignite": 1.3,
        "oil": 1.1,
        "biomass": 0.2,
        "hydro": 0.0,
        "ror": 0.0,
        "run_of_river": 0.0,
        "wind": 0.0,
        "solar": 0.0,
        "pv": 0.0,
        "nuclear": 0.0,
    }

    gen_emission: Dict[int, float] = {}
    renewable_gen_ids: set = set()
    renewable_types = {"wind", "solar", "pv", "hydro", "ror", "run_of_river"}

    for _, row in prods_df.iterrows():
        g_idx = int(row["generator"])
        t_str = str(row["type"]).strip().lower()

        gen_emission[g_idx] = float(base_emissions.get(t_str, 0.5))
        if t_str in renewable_types:
            renewable_gen_ids.add(g_idx)

    # ===============================================================
    # 5) Assemble params
    # ===============================================================
    params = MorlParams(
        econ_scale=econ_scale,
        co2_scale=co2_scale,
        fairness_scale=fairness_scale,
        risk_scale=risk_scale,
        simplicity_scale=simplicity_scale,
        n1_scale=n1_scale,
        redispatch_cost_per_mw=redispatch_cost_per_mw,
        curtailment_cost_per_mw=curtailment_cost_per_mw,
        gen_emission=gen_emission,
        renewable_gen_ids=renewable_gen_ids,
        line_to_zone=line_to_zone,
        gen_to_zone=gen_to_zone,
        zone_population=zone_population,
        critical_lines=None,
    )

    return params



# ===================================================================
# METRIC FUNCTIONS
# ===================================================================

def metric_survival(reward: float) -> float:
    return float(reward)


def metric_longevity(done: bool, info: Dict[str, Any]) -> float:
    """
    Step-wise longevity signal: reward staying alive as long as possible,
    independent of PPO_Reward shaping.

    - +1 for each alive step
    - +B_good at safe episode end (no GAME OVER)
    - -B_bad at GAME OVER
    """
    if not done:
        # still alive -> small positive reward every step
        return 1.0

    # episode ended: inspect exception string
    exc_str = str(info.get("exception", ""))

    # GAME OVER: strongly negative
    if "GAME OVER" in exc_str:
        return -50.0

    # Ended without GAME OVER (natural end of scenario, etc.) -> big bonus
    return 50.0



def metric_risk(next_obs, params: MorlParams) -> float:
    rho = _safe_array(getattr(next_obs, "rho", None))
    if rho is None:
        return 0.0
    return float(np.mean(np.power(rho, 3))) * params.risk_scale


def metric_economic(obs, next_obs, params: MorlParams) -> float:
    """redispatch + curtailment (simple proxy)"""
    prod_prev = _safe_array(getattr(obs, "prod_p", None))
    prod_next = _safe_array(getattr(next_obs, "prod_p", None))

    if prod_prev is None or prod_next is None:
        return 0.0

    delta_p = prod_next - prod_prev
    econ_cost = 0.0

    # Redispatch
    econ_cost += float(np.sum(np.abs(delta_p))) * params.redispatch_cost_per_mw

    # Curtailment (renewables only)
    if params.renewable_gen_ids:
        ids = np.array(list(params.renewable_gen_ids), dtype=int)
        ids = ids[(ids >= 0) & (ids < delta_p.size)]
        if ids.size > 0:
            curtailed = np.sum(np.clip(-delta_p[ids], 0, None))
            econ_cost += float(curtailed) * params.curtailment_cost_per_mw

    return econ_cost / max(params.econ_scale, 1e-6)


def metric_co2_and_renewables(next_obs, params: MorlParams) -> (float, float):
    prod = _safe_array(getattr(next_obs, "prod_p", None))
    if prod is None or prod.size == 0:
        return 0.0, 0.0

    total_p = float(np.sum(np.clip(prod, 0, None)))
    co2 = 0.0
    renewable_ratio = 0.0

    # CO2
    if params.gen_emission:
        for i, p in enumerate(prod):
            if p <= 0:
                continue
            co2 += p * params.gen_emission.get(i, 0.0)
        co2 /= max(params.co2_scale, 1e-6)

    # Renewables
    if params.renewable_gen_ids and total_p > 0:
        ids = np.array(list(params.renewable_gen_ids), dtype=int)
        ids = ids[(ids >= 0) & (ids < prod.size)]
        if ids.size > 0:
            p_ren = float(np.sum(np.clip(prod[ids], 0, None)))
            renewable_ratio = p_ren / total_p

    return co2, renewable_ratio


def metric_fair_rho(next_obs, params: MorlParams) -> float:
    rho = _safe_array(getattr(next_obs, "rho", None))
    zones = params.line_to_zone

    if rho is None or zones is None:
        return 0.0

    # Be robust to tiny mismatches: align to the shorter length
    L = min(rho.size, zones.size)
    rho = rho[:L]
    zones = zones[:L]

    zones = zones.astype(int)
    n_zones = int(np.max(zones) + 1)

    zone_loads = np.zeros(n_zones, float)
    for z in range(n_zones):
        zone_loads[z] = float(np.sum(rho[zones == z]))

    if n_zones > 1:
        return float(np.var(zone_loads)) / max(params.fairness_scale, 1e-6)
    return 0.0


def metric_fair_curtailment(obs, next_obs, params: MorlParams) -> float:
    prod_prev = _safe_array(getattr(obs, "prod_p", None))
    prod_next = _safe_array(getattr(next_obs, "prod_p", None))
    zones = params.gen_to_zone

    if prod_prev is None or prod_next is None or zones is None:
        return 0.0

    if prod_prev.size != zones.size:
        return 0.0

    delta_p = prod_next - prod_prev
    zones = zones.astype(int)
    n_zones = int(np.max(zones) + 1)

    zone_curt = np.zeros(n_zones)
    for i, dp in enumerate(delta_p):
        if dp < 0:
            zone_curt[zones[i]] += -float(dp)

    if n_zones > 1:
        return float(np.var(zone_curt)) / max(params.fairness_scale, 1e-6)
    return 0.0


def metric_equity_curtailment(obs, next_obs, params: MorlParams) -> float:
    prod_prev = _safe_array(getattr(obs, "prod_p", None))
    prod_next = _safe_array(getattr(next_obs, "prod_p", None))
    zones = params.gen_to_zone
    pop = params.zone_population

    if prod_prev is None or prod_next is None or zones is None or pop is None:
        return 0.0

    if prod_prev.size != zones.size:
        return 0.0

    delta_p = prod_next - prod_prev
    zones = zones.astype(int)
    n_zones = int(np.max(zones) + 1)

    zone_curt = np.zeros(n_zones)
    for i, dp in enumerate(delta_p):
        if dp < 0:
            zone_curt[zones[i]] += -float(dp)

    per_capita = []
    for z in range(n_zones):
        pop_z = pop.get(z, 0.0)
        if pop_z > 0:
            per_capita.append(zone_curt[z] / pop_z)

    if len(per_capita) > 1:
        return float(np.var(per_capita)) / max(params.fairness_scale, 1e-6)
    return 0.0


def metric_simplicity(action, params: MorlParams) -> float:
    if action is None:
        return 0.0
    try:
        vect = action.to_vect()
        return float(np.count_nonzero(vect)) * params.simplicity_scale
    except Exception:
        return 0.0


def metric_n1_proxy(next_obs, params: MorlParams) -> float:
    rho = _safe_array(getattr(next_obs, "rho", None))
    if rho is None:
        return 0.0

    if params.critical_lines is not None and len(params.critical_lines) > 0:
        idx = np.array(params.critical_lines)
        idx = idx[(idx >= 0) & (idx < rho.size)]
        if idx.size > 0:
            return float(np.max(rho[idx])) * params.n1_scale

    return float(np.max(rho)) * params.n1_scale

def metric_l2rpn_default(next_obs, params: MorlParams) -> float:
    """
    Reimplementation of the historical L2RPNReward using only `rho`.

    Original idea:
        relative_flow = |flow| / thermal_limit
        x = min(relative_flow, 1)
        score = max(1 - x^2, 0)
        reward = sum(score) / 10000

    Here we use `next_obs.rho` as `relative_flow`.
    """
    rho = _safe_array(getattr(next_obs, "rho", None))
    if rho is None:
        return 0.0

    x = np.minimum(np.abs(rho), 1.0)
    score = np.maximum(1.0 - x**2, 0.0)
    return float(score.sum() / 10000.0)



# ===================================================================
# MASTER FUNCTION
# ===================================================================
def compute_morl_metrics(
    obs,
    next_obs,
    action,
    reward: float,
    done: bool,
    info: Dict[str, Any],
    params: MorlParams,
) -> Dict[str, float]:

    metrics: Dict[str, float] = {}

    metrics["survival"] = metric_survival(reward)
    # New main robustness metric: step-wise longevity
    metrics["longevity"] = metric_longevity(done, info)

    metrics["risk"] = metric_risk(next_obs, params)
    metrics["econ_cost"] = metric_economic(obs, next_obs, params)

    co2, ren = metric_co2_and_renewables(next_obs, params)
    metrics["co2"] = co2
    metrics["renewable_ratio"] = ren

    metrics["fair_rho"] = metric_fair_rho(next_obs, params)
    metrics["fair_curtail"] = metric_fair_curtailment(obs, next_obs, params)
    metrics["equity_curtail"] = metric_equity_curtailment(obs, next_obs, params)

    metrics["simplicity"] = metric_simplicity(action, params)
    metrics["n1_proxy"] = metric_n1_proxy(next_obs, params)

    # NEW: historical L2RPNReward-style score as reference metric
    metrics["l2rpn_reward"] = metric_l2rpn_default(next_obs, params)


    #transformed metrics to prepare them for potential MORL ussage
    metrics["transformed_survival"] = (metrics["survival"] - 1.1) / 0.1
    metrics["transformed_risk"] = (metrics["risk"] - 0.08) / 0.03
    metrics["transformed_econ_cost"] = (metrics["econ_cost"] - 5.0) / 1.0
    metrics["transformed_co2"] = (metrics["co2"] - 70.0) / 7.0
    metrics["transformed_renewable_ratio"] = (metrics["renewable_ratio"] - 0.65) / 0.02
    metrics["transformed_fair_rho"] = (metrics["fair_rho"] - 20.0) / 3.0
    metrics["transformed_fair_curtail"] = (metrics["fair_curtail"] - 22.0) / 10.0
    metrics["transformed_equity_curtail"] = (metrics["equity_curtail"] * 1e5 - 4.0) / 1.0
    metrics["transformed_simplicity"] = (metrics["simplicity"] - 0.18) / 0.05
    metrics["transformed_n1_proxy"] = (metrics["n1_proxy"] - 0.75) / 0.02
    metrics["transformed_l2rpn_reward"] = (metrics["l2rpn_reward"] - 0.0051) / 0.0002
    metrics["transformed_longevity"] = (metrics["longevity"] - 1.0) / 5.0

    return metrics

def build_gated_scalar_reward(metrics: Dict[str, float], morl_weights ={
    "w_fair_rho": 1.0,"w_fair_curt": 1.0,"w_equity": 1.0, "w_ren": 1.0,"w_co2": 1.0,
    "w_risk": 1.0,"w_n1": 1.0,"w_econ": 1.0,"w_simplicity": 1.0,"w_l2rpn": 1.0,
    "tau_primary": 0.5,"alpha_fair": 0.1,
    "alpha_sust": 0.1,
    "alpha_struct": 0.3
  } ) -> Dict[str, float]:
    """
    Build a single scalar reward from transformed MORL metrics using a
    gated / tiered scheme.

    - Primary block: transformed_survival
    - Fairness block: fair_rho, fair_curtail, equity_curtail
    - Sustainability block: renewable_ratio, co2
    - Structural block: risk, n1_proxy, econ_cost, simplicity, l2rpn_reward

    Returns a dict with:
      - scalar_reward: final reward used for PPO
      - primary_block, fairness_block, sustain_block, structural_block
    so that they can be logged separately.
    """

    # --- 1) PRIMARY block: survival only ---
    primary_block = metrics.get(
        "transformed_survival",
        metrics.get("transformed_survival", 0.0),
    )

    # --- 2) FAIRNESS block (lower variance => better, so minus signs) ---
    w_fair_rho = morl_weights.get("w_fair_rho", 1.0)
    w_fair_curt = morl_weights.get("w_fair_curt", 1.0)
    w_equity = morl_weights.get("w_equity", 1.0)

    fairness_block = (
        -w_fair_rho * metrics.get("transformed_fair_rho", 0.0)
        -w_fair_curt * metrics.get("transformed_fair_curtail", 0.0)
        -w_equity * metrics.get("transformed_equity_curtail", 0.0)
    )

    # --- 3) SUSTAINABILITY block ---
    w_ren = morl_weights.get("w_ren", 1.0)
    w_co2 = morl_weights.get("w_co2", 1.0)

    sustain_block = (
        +w_ren * metrics.get("transformed_renewable_ratio", 0.0)
        -w_co2 * metrics.get("transformed_co2", 0.0)
    )

    # --- 4) STRUCTURAL block (grid security / efficiency signals) ---
    w_risk = morl_weights.get("w_risk", 1.0)
    w_n1 = morl_weights.get("w_n1", 1.0)
    w_econ = morl_weights.get("w_econ", 1.0)
    w_simplicity = morl_weights.get("w_simplicity", 1.0)
    w_l2rpn = morl_weights.get("w_l2rpn", 1.0)

    structural_block = (
        -w_risk * metrics.get("transformed_risk", 0.0)
        +w_n1 * metrics.get("transformed_n1_proxy", 0.0)
        -w_econ * metrics.get("transformed_econ_cost", 0.0)
        +w_simplicity * metrics.get("transformed_simplicity", 0.0)
        +w_l2rpn * metrics.get("transformed_l2rpn_reward", 0.0)
    )

    # --- 5) GATE: only add secondary blocks once primary is "good enough" ---
    # Threshold in *transformed* units. 0.0 ≈ baseline survival.
    tau_primary = morl_weights.get("tau_primary", 0.5)

    # How much secondary blocks matter once above threshold
    alpha_fair = morl_weights.get("alpha_fair", 0.1)
    alpha_sust = morl_weights.get("alpha_sust", 0.1)
    alpha_struct = morl_weights.get("alpha_struct", 0.3)

    if primary_block < tau_primary:
        # Not yet at baseline survival -> focus purely on staying alive.
        scalar_reward = primary_block
    else:
        # Survival is okay -> refine with fairness / sustainability / structure.
        scalar_reward = (
            primary_block
            + alpha_fair * fairness_block
            + alpha_sust * sustain_block
            + alpha_struct * structural_block
        )

    return {
        "scalar_reward": scalar_reward,
        "primary_block": primary_block,
        "fairness_block": fairness_block,
        "sustain_block": sustain_block,
        "structural_block": structural_block,
    }


# ===================================================================
# MAIN TEST (run this file directly)
# ===================================================================
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        message='There were some Nan in the pp_net.trafo["tap_step_degree"]',
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="We found either some slack coefficient to be < 0. or they were all 0.",
        category=UserWarning,
    )

    print("=== MORL Metrics Quick Test ===")

    try:
        # If you have the NeurIPS dataset locally, this is closer to your actual setup.
        DATA_PATH = "./training_data_track1"
        SCENARIO_PATH = './training_data_track1/chronics'
        from lightsim2grid import LightSimBackend
        import grid2op
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH,
                           chronics_path=SCENARIO_PATH,
                           backend=backend)
        params = build_morl_params_from_dataset(DATA_PATH)
        print("Loaded env from local dataset:", DATA_PATH)
    except Exception:
        # Fallback: use the registered name (smaller built-in version).
        SCENARIO_PATH = './training_data_track1/chronics'

        from lightsim2grid import LightSimBackend
        import grid2op

        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH,
                           chronics_path=SCENARIO_PATH,
                           backend=backend)
        params = MorlParams()
        print("Loaded fallback env: l2rpn_neurips_2020_track1_small")

    obs = env.reset()
    for step in range(5):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        metrics = compute_morl_metrics(obs, next_obs, action, reward, done, info, params)

        print(f"\n--- Step {step} ---")
        for k, v in metrics.items():
            print(f"{k:20s}: {v}")

        obs = next_obs
        if done:
            obs = env.reset()
