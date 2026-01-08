#!/usr/bin/env python3 

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Distance archetypes
def d_log_ratio(cand: pd.Series, ref: pd.Series, *, eps: float = 0.0) -> pd.Series:
    """Compute absolute log-ratio distance for positive, scale-dependent values.

    Parameters
    ----------
    cand : pandas.Series
        Candidate values.
    ref : pandas.Series
        Reference values.
    eps : float, optional
        Small constant added to avoid division by zero.

    Returns
    -------
    pandas.Series
        Absolute log-ratio distances.
    """
    c = cand.to_numpy(dtype=float, copy=False)
    r = ref.to_numpy(dtype=float, copy=False)
    if eps:
        c = c + eps
        r = r + eps
    out = np.abs(np.log(c / r))
    return pd.Series(out, index=cand.index)


def d_abs_diff(cand: pd.Series, ref: pd.Series) -> pd.Series:
    """Compute absolute differences between candidate and reference values.

    Parameters
    ----------
    cand : pandas.Series
        Candidate values.
    ref : pandas.Series
        Reference values.

    Returns
    -------
    pandas.Series
        Absolute differences.
    """
    out = np.abs(cand.to_numpy(float, copy=False) - ref.to_numpy(float, copy=False))
    return pd.Series(out, index=cand.index)


def d_one_minus_similarity01(cand: pd.Series, ref: Optional[pd.Series] = None) -> pd.Series:
    """Convert a [0, 1] similarity score into a distance.

    Parameters
    ----------
    cand : pandas.Series
        Similarity values in [0, 1].
    ref : pandas.Series, optional
        Unused placeholder for a reference series.

    Returns
    -------
    pandas.Series
        Distances computed as ``1 - similarity``.
    """
    out = 1.0 - cand.to_numpy(dtype=float, copy=False)
    return pd.Series(out, index=cand.index)


def d_spatial_scaled(cand_dist: pd.Series, ref: Optional[pd.Series] = None, *, scale_m: float) -> pd.Series:
    """Compute spatial mismatch distance scaled by a reference distance.

    Parameters
    ----------
    cand_dist : pandas.Series
        Candidate distances in meters.
    ref : pandas.Series, optional
        Unused placeholder for a reference series.
    scale_m : float
        Distance scale where the penalty is approximately 1.

    Returns
    -------
    pandas.Series
        Scaled distances (dimensionless).
    """
    d = cand_dist.to_numpy(dtype=float, copy=False)
    out = d / float(scale_m)
    return pd.Series(out, index=cand_dist.index)


DIST_ARCHETYPES: Dict[str, Callable[..., pd.Series]] = {
    "scale": d_log_ratio,               # area, qbar, etc.
    "abs": d_abs_diff,                  # absolute units
    "sim01": d_one_minus_similarity01,  # bounded similarity -> distance
    "spatial": d_spatial_scaled,        # gauge->candidate distance (meters) scaled
}


# Specs + schemas
@dataclass(frozen=True)
class DistanceSpec:
    """Specification for a single distance component."""
    name: str
    cand_col: str
    ref_col: Optional[str] = None
    weight: float = 1.0
    dist_fn: Callable[..., pd.Series] = d_log_ratio
    kwargs: Mapping[str, Any] = None  # extra args to dist_fn


def make_spec_from_plan_item(item: Mapping[str, Any]) -> DistanceSpec:
    """Build a DistanceSpec from a distance-plan item.

    Parameters
    ----------
    item : Mapping[str, Any]
        Distance component definition. Typical keys include ``attr``, ``name``,
        ``dist``, ``weight``, ``cand_col``, and ``ref_col``.

    Returns
    -------
    DistanceSpec
        Parsed distance specification.
    """
    attr = item.get("attr")
    if not attr:
        raise ValueError("distance_plan item must include 'attr'")

    name = item.get("name", attr)
    dist_key = item.get("dist", "scale")
    weight = float(item.get("weight", 1.0))
    cand_col = item.get("cand_col") #or product_schema.col(attr)

    dist_fn = DIST_ARCHETYPES[dist_key]

    # ref_col only needed for some archetypes
    if dist_key in ("sim01", "spatial"):
        ref_col = None
    else:
        ref_col = item.get("ref_col") #or reference_schema.col(attr)

    # remaining kwargs to dist_fn
    reserved = {"attr", "name", "dist", "weight", "cand_col", "ref_col"}
    kwargs = {k: v for k, v in item.items() if k not in reserved}

    return DistanceSpec(
        name=name,
        cand_col=cand_col,
        ref_col=ref_col,
        weight=weight,
        dist_fn=dist_fn,
        kwargs=kwargs or None,
    )


def get_gauge_dist_spec(scale_m: float, weight: float = 1.0):
    return make_spec_from_plan_item({
        'attr': 'gauge_dist', 
        'dist': 'spatial', 
        'weight': weight, 
        'ref_col': None, 
        'cand_col': 'distance_m', 
        'kwargs': {'scale_m': scale_m}
    })


def get_catchment_area_spec(dataset: str, reference_col: str, weight: float = 1.0):
    if dataset.upper() == 'GRIT':
        cand_col = 'drainage_area_out'

    return make_spec_from_plan_item({
        'attr': 'drainage_area', 
        'dist': 'scale', 
        'weight': weight,
        'ref_col': reference_col, 
        'cand_col': cand_col
    })


# def get_mean_annual_flow_spec(dataset: str, reference_col: str, weight: float = 1.0):
#     if dataset.upper() == 'GRIT':
#         cand_col = 'mean_annual_flow'
#     return make_spec_from_plan_item({
#         'attr': 'mean_annual_flow', 
#         'dist': 'scale', 
#         'weight': weight,
#         'ref_col': reference_col, 
#         'cand_col': cand_col
#     })


def compute_candidate_distances_from_plan(
    candidates: pd.DataFrame,
    *,
    # distance_plan: Sequence[Mapping[str, Any]],
    distance_plan: Sequence[DistanceSpec],
    aggregation_method: str = "weighted_mean",     # "weighted_sum" or "weighted_mean"
    require_any: bool = True,       # True: need at least one component present per row
    drop_missing_components: bool = True,
) -> pd.DataFrame: 
    """Compute component distances and aggregate them per candidate.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Candidate rows containing the required component columns.
    distance_plan : Sequence[Mapping[str, Any]]
        Sequence of distance component specifications.
    aggregation_method : str, optional
        "weighted_sum" or "weighted_mean".
    require_any : bool, optional
        If True, each row must have at least one component present.
    drop_missing_components : bool, optional
        If True, skip components missing data/columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with per-component distance columns and a total ``distance``.
    """
    # Work on a copy
    df = candidates.copy()

    # Build specs
    specs: List[DistanceSpec] = []
    for item in distance_plan:
        specs.append(item)
        # specs.append(make_spec_from_plan_item(item))

    # Determine which reference columns are needed
    needed_ref_cols = sorted({s.ref_col for s in specs if s.ref_col is not None})
    missing_ref = [c for c in needed_ref_cols if c not in df.columns]
    if missing_ref and not drop_missing_components:
        raise KeyError(f"Reference missing columns: {missing_ref}")

    # Compute component distances
    component_dist_cols: List[str] = []
    weights: Dict[str, float] = {}
    for s in specs:
        dcol = f"d_{s.name}"
        if s.cand_col not in df.columns:
            reason = f"missing candidate column '{s.cand_col}'"
            if drop_missing_components:
                continue
            raise KeyError(reason)

        cand_series = df[s.cand_col]
        ref_series = df[s.ref_col] if s.ref_col is not None else None
        kwargs = dict(s.kwargs or {})
        try:
            if ref_series is None:
                df[dcol] = s.dist_fn(cand_series, None, **kwargs)
            else:
                df[dcol] = s.dist_fn(cand_series, ref_series, **kwargs)
        except TypeError:
            # Fallback if there are unsupported kwargs
            df[dcol] = s.dist_fn(cand_series, ref_series)  # type: ignore[misc]

        component_dist_cols.append(dcol)
        weights[s.name] = float(s.weight)

    # Aggregate distance
    if not component_dist_cols:
        df["distance"] = np.nan
        return df

    D = df[component_dist_cols].to_numpy(dtype=float, copy=False)
    W = np.asarray([weights[name.replace("d_", "")] for name in component_dist_cols], dtype=float)
    valid = ~np.isnan(D)
    ok = valid.any(axis=1) if require_any else valid.all(axis=1)
    weighted = np.where(valid, D * W, np.nan)
    num = np.nansum(weighted, axis=1)

    if aggregation_method == "weighted_sum":
        dist = num
    elif aggregation_method == "weighted_mean":
        denom = np.nansum(np.where(valid, W, np.nan), axis=1) # i.e. sum of weights for valid components
        dist = num / denom
    else:
        raise ValueError(f"Unknown aggregation method '{aggregation_method}'")

    dist = np.where(ok, dist, np.nan)
    df["distance"] = dist
    return df
