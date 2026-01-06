#!/usr/bin/env python3 

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Distance archetypes
def d_log_ratio(cand: pd.Series, ref: pd.Series, *, eps: float = 0.0) -> pd.Series:
    """|log((cand+eps)/(ref+eps))| for strictly-positive, scale-dependent vars (area, mean flow, etc.)."""
    c = cand.to_numpy(dtype=float, copy=False)
    r = ref.to_numpy(dtype=float, copy=False)
    if eps:
        c = c + eps
        r = r + eps
    out = np.abs(np.log(c / r))
    return pd.Series(out, index=cand.index)

# def d_percent_error(cand: pd.Series, ref: pd.Series, *, eps: float = 1e-12) -> pd.Series:
#     """|cand-ref| / max(|ref|, eps). Not bounded; it's a distance in relative units."""
#     c = cand.to_numpy(dtype=float, copy=False)
#     r = ref.to_numpy(dtype=float, copy=False)
#     denom = np.maximum(np.abs(r), eps)
#     out = np.abs(c - r) / denom
#     return pd.Series(out, index=cand.index)


def d_abs_diff(cand: pd.Series, ref: pd.Series) -> pd.Series:
    """|cand-ref| for vars already commensurate."""
    out = np.abs(cand.to_numpy(float, copy=False) - ref.to_numpy(float, copy=False))
    return pd.Series(out, index=cand.index)


def d_one_minus_similarity01(cand: pd.Series, ref: Optional[pd.Series] = None) -> pd.Series:
    """If cand is already a similarity in [0,1], distance = 1 - similarity."""
    out = 1.0 - cand.to_numpy(dtype=float, copy=False)
    return pd.Series(out, index=cand.index)


def d_spatial_scaled(cand_dist: pd.Series, ref: Optional[pd.Series] = None, *, scale_m: float = 1000.0) -> pd.Series:
    """
    Spatial mismatch distance: dist_m / scale_m (dimensionless).
    Choose scale_m as the distance where you feel the penalty should be ~1.
    """
    d = cand_dist.to_numpy(dtype=float, copy=False)
    out = d / float(scale_m)
    return pd.Series(out, index=cand_dist.index)


DIST_ARCHETYPES: Dict[str, Callable[..., pd.Series]] = {
    "scale": d_log_ratio,               # area, qbar, etc.
    # "pct": d_percent_error,             # relative error as distance
    "abs": d_abs_diff,                  # absolute units
    "sim01": d_one_minus_similarity01,  # bounded similarity -> distance
    "spatial": d_spatial_scaled,        # gauge->candidate distance (meters) scaled
}


# Specs + schemas
@dataclass(frozen=True)
class DistanceSpec:
    name: str
    cand_col: str
    ref_col: Optional[str] = None
    weight: float = 1.0
    dist_fn: Callable[..., pd.Series] = d_log_ratio
    kwargs: Mapping[str, Any] = None  # extra args to dist_fn


# @dataclass(frozen=True)
# class ProductSchema:
#     """Maps logical attribute names -> candidate df column names."""
#     product: str
#     cand_cols: Mapping[str, str]

#     def col(self, attr: str) -> str:
#         if attr not in self.cand_cols:
#             raise KeyError(f"{self.product} schema has no candidate mapping for '{attr}'")
#         return self.cand_cols[attr]

# @dataclass(frozen=True)
# class ReferenceSchema:
#     """Maps logical attribute names -> reference df column names."""
#     ref_cols: Mapping[str, str]

#     def col(self, attr: str) -> str:
#         if attr not in self.ref_cols:
#             raise KeyError(f"Reference schema has no mapping for '{attr}'")
#         return self.ref_cols[attr]


def make_spec_from_plan_item(
    item: Mapping[str, Any],
    # *,
    # product_schema: ProductSchema,
    # reference_schema: ReferenceSchema,
) -> DistanceSpec:
    """
    item fields (typical):
      - name: output component name (defaults to attr)
      - attr: logical attribute name (e.g., "area", "qbar", "gauge_dist_m")
      - dist: archetype key in DIST_ARCHETYPES (e.g., "scale", "spatial", "sim01")
      - weight: float
      - cand_col: optional override (bypass schema)
      - ref_col: optional override (bypass reference schema)
      - ...: any kwargs for the archetype (e.g., eps, scale_m)
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

def _resolve_ref_column(
    df: pd.DataFrame,
    ref: Optional[pd.DataFrame],
    *,
    site_col: Optional[str],
    ref_col: str,
    drop_missing: bool,
) -> Optional[pd.Series]:
    """
    Return a Series aligned to df rows, or None if unavailable.
    """
    # Case 1: already present in candidates
    if ref_col in df.columns:
        return df[ref_col]

    # Case 2: lookup from reference table
    if ref is not None and site_col is not None:
        if site_col not in df.columns:
            raise KeyError(f"site_col='{site_col}' not in candidates")
        if ref_col not in ref.columns:
            if drop_missing:
                return None
            raise KeyError(f"reference missing '{ref_col}'")
        return df[site_col].map(ref[ref_col])

    # Case 3: nowhere to get it
    if drop_missing:
        return None
    raise KeyError(f"Cannot resolve reference column '{ref_col}'")

def compute_candidate_distances_from_plan(
    candidates: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
    *,
    site_col: Optional[str] = None,
    distance_plan: Sequence[Mapping[str, Any]],
    aggregation_method: str = "weighted_mean",     # "weighted_sum" or "weighted_mean"
    require_any: bool = True,       # True: need at least one component present per row
    drop_missing_components: bool = True,
) -> pd.DataFrame: 
    """Compute component distances and aggregate them per candidate.

    Parameters
    ----------
        candidates : pandas.DataFrame
            Candidate rows containing the required component columns.
        reference : pandas.DataFrame, optional
            Optional reference table used to resolve reference columns.
        site_col : str, optional
            Column in candidates used to map reference values.
        distance_plan : List[Mapping[str, Any]]
            Sequence of distance component specifications.
        aggregation_method : str, optional
            "weighted_sum" or "weighted_mean".
        require_any : bool, optional
            If True, each row must have at least one component present.
        drop_missing_components : bool, optional
            If True, skip components missing data/columns.

    Returns:
    --------
        DataFrame with per-component distance columns and a total 'distance'.

    """
    df = candidates.copy()

    # Build specs
    specs: List[DistanceSpec] = []
    for item in distance_plan:
        specs.append(make_spec_from_plan_item(item))

    # # Determine which ref columns are needed
    # needed_ref_cols = sorted({s.ref_col for s in specs if s.ref_col is not None})
    # missing_ref = [c for c in needed_ref_cols if c not in reference.columns]
    # if missing_ref and not drop_missing_components:
    #     raise KeyError(f"Reference missing columns: {missing_ref}")

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

        ref_series = None
        if s.ref_col is not None:
            ref_series = _resolve_ref_column(
                df,
                reference,
                site_col=site_col,
                ref_col=s.ref_col,
                drop_missing=drop_missing_components,
            )
            if ref_series is None:
                continue

        cand_series = df[s.cand_col]
        ref_series = df[s.ref_col] if s.ref_col is not None else None
        kwargs = dict(s.kwargs or {})

        try:
            if ref_series is None:
                df[dcol] = s.dist_fn(cand_series, None, **kwargs)
            else:
                df[dcol] = s.dist_fn(cand_series, ref_series, **kwargs)
        except TypeError:
            # fallback if dist_fn expects (cand, ref) only
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
        denom = np.nansum(np.where(valid, W, np.nan), axis=1)
        dist = num / denom
    else:
        raise ValueError(f"Unknown agg='{aggregation_method}'")
    dist = np.where(ok, dist, np.nan)
    df["distance"] = dist
    return df

