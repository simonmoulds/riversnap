from __future__ import annotations

from dataclasses import dataclass
from abc import ABC
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class HydrographySchema(ABC):
    """
    Base class describing how a particular river hydrography product stores
    candidate reach attributes in a DataFrame.

    Subclasses should override the class attributes (column name strings).
    """
    product: str
    # Reach ID
    id_col: str
    # Candidate drainage area (e.g., upstream area at candidate location)
    drainage_area_col: str
    # Optional but commonly used candidate attributes
    mean_annual_flow_col: Optional[str] = None

    def required_columns(self) -> Sequence[str]:
        """Columns that *must* be present in a candidate DataFrame."""
        return [self.id_col, self.drainage_area_col]

    def optional_columns(self) -> Sequence[str]:
        """Columns that are nice to have (if you intend to use them)."""
        cols = [
            self.mean_annual_flow_col,
        ]
        return [c for c in cols if c is not None]

    def as_mapping(self) -> Dict[str, str]:
        """
        Map logical attribute keys -> column names.
        Useful for config-driven DistanceSpec building.
        """
        mapping = {
            "reach_id": self.id_col,
            "area": self.drainage_area_col,
        }
        if self.mean_annual_flow_col:
            mapping["qmean"] = self.mean_annual_flow_col
        return mapping

    def col(self, key: str) -> str:
        """Get the column name for a logical key."""
        mapping = self.as_mapping()
        if key not in mapping:
            raise KeyError(f"{self.product}: schema has no mapping for key '{key}'")
        return mapping[key]

    def validate_candidates_df(self, df: pd.DataFrame, *, extra_required: Iterable[str] = ()) -> None:
        """
        Validate that required columns are present.
        `extra_required` can be used to enforce columns implied by a distance plan.
        """
        required = set(self.required_columns()) | set(extra_required)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{self.product}: candidate DataFrame missing columns: {missing}")


@dataclass(frozen=True)
class GRITSchema(HydrographySchema):
    """
    Example schema for GRIT (placeholder column names — replace with actual names).
    """
    product: str = "GRIT"
    id_col: str = "global_id"
    drainage_area_col: str = "uparea_km2"

    # optional fields if you have them
    mean_annual_flow_col: Optional[str] = None