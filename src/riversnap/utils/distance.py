#!/usr/bin/env python3 

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import inspect
import numpy as np
import pandas as pd

EPS = 1e-12

# Distance helpers
# def d_log_ratio(cand: pd.Series, ref: pd.Series, *, eps: float = 0.0) -> pd.Series:
def d_log_ratio(cand: pd.Series, ref: pd.Series) -> pd.Series:
    """Compute absolute log-ratio distance.

    The log-ratio distance is suitable for strictly positive, scale-dependent 
    variables for which relative differences are more meaningful than absolute 
    differences. It is particularly appropriate for attributes such as drainage 
    area or mean annual discharge, which often span several orders of magnitude 
    and for which proportional over- and under-estimation should be penalised 
    symmetrically.

    Note that if d = |log(cand / ref)| then the percentage error can be computed 
    as (exp(d) - 1) * 100%.

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
    if EPS:
        c = c + EPS
        r = r + EPS
    out = np.abs(np.log(c / r))
    return pd.Series(out, index=cand.index)


def d_abs_diff(cand: pd.Series, ref: pd.Series) -> pd.Series:
    """Compute absolute differences between candidate and reference values.

    The absolute difference distance is suitable for variables that are already 
    expressed on a comparable scale and for which deviations are naturally 
    interpreted in absolute units. This formulation is appropriate for bounded 
    indices (e.g. baseflow index), quantities with limited dynamic range, or 
    variables where ratio-based measures are not meaningful.

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

    This distance is suitable for variables that are already expressed as 
    similarity scores bounded between 0 and 1, typically derived from 
    independent analyses or matching algorithms. Converting similarity to 
    distance using a simple linear transformation enables such externally 
    defined metrics to be integrated directly into the distance-based 
    snapping framework.

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


@dataclass(frozen=True)
class DiagnosticContext:
    name: str
    cand: pd.Series
    ref: Optional[pd.Series]
    d: Optional[pd.Series]   # component distance d_<name>


# @dataclass(frozen=True)
# class DistanceSpec:
#     """Specification for a single distance component."""
#     name: str
#     cand_col: str
#     dist_fn: Callable[..., pd.Series]
#     ref_col: Optional[str] = None
#     weight: float = 1.0
#     diagnostics: Sequence[str] = ()
#     kwargs: Mapping[str, Any] = None  # extra args to dist_fn
class DistanceSpec:
    """
    Specification for a single distance component used in snapping.
    """
    def __init__(
        self,
        *,
        name: str,
        dist_fn: str,
        cand_col: str,
        ref_col: str | None = None,
        weight: float = 1.0,
        diagnostics: tuple[str, ...] | None = None,
        **kwargs,
    ):
        self.name = str(name)

        # Resolve distance function archetype
        if dist_fn not in DIST_ARCHETYPES:
            raise ValueError(
                f"Unknown distance function archetype '{dist_fn}' "
                f"for component '{self.name}'"
            )
        self.dist_key = dist_fn
        self.dist_fn = DIST_ARCHETYPES[dist_fn]

        self.cand_col = str(cand_col)
        self.ref_col = ref_col
        self.weight = float(weight)

        # Diagnostics: user-specified or defaults by archetype
        if diagnostics is None:
            diagnostics = DEFAULT_DIAGNOSTICS.get(self.dist_key, ())
        self.diagnostics = tuple(diagnostics)

        # Remaining kwargs passed to dist_fn
        self.kwargs = dict(kwargs) if kwargs else {}

        # Optional: validate diagnostics early
        self._validate_diagnostics()

    def _validate_diagnostics(self) -> None:
        for d in self.diagnostics:
            if d not in DIAGNOSTICS:
                raise ValueError(
                    f"Unknown diagnostic '{d}' for component '{self.name}'"
                )

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> "DistanceSpec":
        """
        Convenience constructor for config-driven workflows.
        """
        required = {"name", "dist_fn", "cand_col"}
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Distance plan item missing required keys: {sorted(missing)}")

        return cls(**item)


DiagFn = Callable[[DiagnosticContext], pd.Series]


def diag_m(ctx: DiagnosticContext) -> pd.Series:
    return ctx.cand.astype(float)


def diag_pp(ctx: DiagnosticContext) -> pd.Series:
    if ctx.ref is None:
        return pd.Series(np.nan, index=ctx.cand.index)
    return 100.0 * (ctx.cand.astype(float) - ctx.ref.astype(float)).abs()


def diag_pct(ctx: DiagnosticContext) -> pd.Series:
    # expects ctx.d from log-ratio
    if ctx.d is None:
        return pd.Series(np.nan, index=ctx.cand.index)
    return (np.exp(ctx.d.astype(float)) - 1.0) * 100.0


def diag_factor(ctx: DiagnosticContext) -> pd.Series:
    if ctx.d is None:
        return pd.Series(np.nan, index=ctx.cand.index)
    return np.exp(ctx.d.astype(float))


def diag_err(ctx: DiagnosticContext) -> pd.Series:
    if ctx.ref is None:
        return pd.Series(np.nan, index=ctx.cand.index)
    return (ctx.cand.astype(float) - ctx.ref.astype(float)).abs()


DIST_ARCHETYPES: Dict[str, Callable[..., pd.Series]] = {
    "scale": d_log_ratio,
    "abs": d_abs_diff,
    "spatial": d_spatial_scaled,
    "sim01": d_one_minus_similarity01,
}


# Default diagnostics per archetype
DEFAULT_DIAGNOSTICS: Dict[str, Sequence[str]] = {
    "spatial": ("m",),              # raw metres
    "scale": ("pct", "factor"),     # interpret log-ratio in human terms
    "abs": ("err",),                # native-unit error
    "sim01": ("sim",),              # echo similarity score
}


DIAGNOSTICS: dict[str, DiagFn] = {
    "m": diag_m,
    "pp": diag_pp,
    "pct": diag_pct,
    "factor": diag_factor,
    "err": diag_err,
}


# def make_distance_spec_from_dict(item: Mapping[str, Any]) -> DistanceSpec:
#     """Build a DistanceSpec from a distance-plan item.

#     Parameters
#     ----------
#     item : Mapping[str, Any]
#         Distance component definition. 

#     Returns
#     -------
#     DistanceSpec
#         Parsed distance specification.
#     """
#     # Check required keys are present
#     required_keys = {"name", "dist_fn", "cand_col"}
#     missing = [k for k in required_keys if k not in item]
#     if missing:
#         raise ValueError(f"Distance plan item missing required keys: {missing}")

#     # Parse fields, making sure distance function is callable
#     name = str(item["name"])
#     dist_fn = str(item["dist_fn"])
#     if dist_fn not in DIST_ARCHETYPES:
#         raise ValueError(f"Unknown distance function archetype '{dist_fn}' for component '{name}'")
#     # dist_fn = DIST_ARCHETYPES[item["dist_fn"]]
#     # if not inspect.isfunction(dist_fn):
#     #     raise ValueError(f"Distance function for '{name}' is not callable")
#     diags = item.get("diagnostics", DEFAULT_DIAGNOSTICS.get(dist_fn, ()))

#     # TODO check diagnostics 

#     weight = float(item.get("weight", 1.0))
#     cand_col = str(item["cand_col"])
#     ref_col = item.get("ref_col", None)

#     # Remaining kwargs to dist_fn
#     reserved = {"name", "dist_fn", "weight", "cand_col", "ref_col"}
#     kwargs = {k: v for k, v in item.items() if k not in reserved}

#     return DistanceSpec(
#         name=name,
#         cand_col=cand_col,
#         ref_col=ref_col,
#         weight=weight,
#         dist_fn=dist_fn,
#         diagnostics=diags,
#         kwargs=kwargs or None,
#     )


def get_gauge_dist_spec(scale_m: float, weight: float = 1.0):
    """Helper function to build a distance spec for gauge-to-candidate distance.

    Parameters
    ----------
    scale_m : float
        Distance scale where the penalty is approximately 1.
    weight : float, optional
        Weight applied to this component.

    Returns
    -------
    DistanceSpec
        Distance specification for gauge distance.
    """
    return DistanceSpec.from_dict({
        'name': 'gauge_dist', 
        'ref_col': None, 
        'cand_col': 'distance_m', 
        'dist_fn': "spatial", 
        'weight': weight, 
        'scale_m': scale_m
    })


def get_catchment_area_spec(dataset: str, reference_col: str, weight: float = 1.0):
    """Helper function to build a distance spec for catchment area comparison.

    Parameters
    ----------
    dataset : str
        Dataset name used to resolve candidate column names.
    reference_col : str
        Reference column name in the candidate table.
    weight : float, optional
        Weight applied to this component.

    Returns
    -------
    DistanceSpec
        Distance specification for catchment area.
    """
    if dataset.upper() == 'GRIT':
        cand_col = 'drainage_area_out'

    # return make_distance_spec_from_dict({
    return DistanceSpec.from_dict({
        'name': 'drainage_area', 
        'ref_col': reference_col, 
        'cand_col': cand_col,
        'dist_fn': "scale", 
        'weight': weight,
    })


@dataclass(frozen=True)
class DistanceReport:
    used: List[str]                     # component names used
    skipped: List[Tuple[str, str]]      # (component name, reason)
    attribute_cols: List[str]
    distance_component_cols: List[str]  # d_<name> columns produced
    weights: Dict[str, float]           # name -> weight
    diagnostic_cols: List[str]
    distance: List[str]


def compute_candidate_distances_from_plan(
    candidates: pd.DataFrame,
    *,
    specs: Sequence[DistanceSpec],
    aggregation_method: str = "weighted_mean",     # "weighted_sum" or "weighted_mean"
    require_any: bool = True,       # True: need at least one component present per row
    drop_missing_components: bool = True,
) -> pd.DataFrame: 
    """Compute component distances and aggregate them per candidate.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Candidate rows containing the required component columns.
    specs : Sequence[Mapping[str, Any]]
        Sequence of distance component specifications.
    aggregation_method : str, optional
        Method to merge all distances into a single value. Either 
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

    # Determine which reference columns are needed
    needed_ref_cols = sorted({s.ref_col for s in specs if s.ref_col is not None})
    missing_ref = [c for c in needed_ref_cols if c not in df.columns]
    if missing_ref and not drop_missing_components:
        raise KeyError(f"Reference missing columns: {missing_ref}")

    # Compute component distances
    used: List[str] = [] 
    skipped: List[str] = []
    attribute_cols: List[str] = []
    distance_component_cols: List[str] = []
    weights: Dict[str, float] = {}
    diagnostic_cols: List[str] = []

    for s in specs:
        dcol = f"d_{s.name}"
        if s.cand_col not in df.columns:
            reason = f"missing candidate column '{s.cand_col}'"
            if drop_missing_components:
                skipped.append(s.name)
                continue
            raise KeyError(reason)

        cand_series = df[s.cand_col]
        ref_series = df[s.ref_col] if s.ref_col is not None else None
        kwargs = dict(s.kwargs or {})

        # dist_fn = DIST_ARCHETYPES[s.dist_fn]
        try:
            if ref_series is None:
                df[dcol] = s.dist_fn(cand_series, None, **kwargs)
            else:
                df[dcol] = s.dist_fn(cand_series, ref_series, **kwargs)
        except TypeError:
            # Fallback if there are unsupported kwargs
            df[dcol] = dist_fn(cand_series, ref_series)  # type: ignore[misc]

        used.append(s.name)
        if s.ref_col:
            attribute_cols.append(s.ref_col)
        attribute_cols.append(s.cand_col)
        distance_component_cols.append(dcol)
        weights[s.name] = float(s.weight)

        # Compute diagnostic columns 
        ctx = DiagnosticContext(name=s.name, cand=cand_series, ref=ref_series, d=df[dcol])
        for key in s.diagnostics:
            fn = DIAGNOSTICS[key]
            diagcol = f"{key}_{s.name}"
            df[diagcol] = fn(ctx)
            diagnostic_cols.append(diagcol)

    # Aggregate distance
    if not distance_component_cols:
        df["distance"] = np.nan
        return df

    D = df[distance_component_cols].to_numpy(dtype=float, copy=False)
    W = np.asarray([weights[name.replace("d_", "")] for name in distance_component_cols], dtype=float)
    valid = ~np.isnan(D)
    ok = valid.any(axis=1) if require_any else valid.all(axis=1)
    weighted = np.where(valid, D * W, np.nan)
    num = np.nansum(weighted, axis=1)

    if aggregation_method == "weighted_sum":
        dist = num
    elif aggregation_method == "weighted_mean":
        denom = np.nansum(np.where(valid, W, np.nan), axis=1) # i.e. sum of weights for valid components
        if EPS: 
            num = num + EPS 
            denom = denom + EPS 
        dist = num / denom
    else:
        raise ValueError(f"Unknown aggregation method '{aggregation_method}'")

    dist = np.where(ok, dist, np.nan)
    df["distance"] = dist

    report = DistanceReport(
        used=used,
        skipped=skipped,
        attribute_cols=attribute_cols,
        distance_component_cols=distance_component_cols,
        weights=weights,
        diagnostic_cols=diagnostic_cols,
        distance=['distance']
    )
    return df, report
