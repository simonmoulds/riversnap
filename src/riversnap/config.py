EPS_DEFAULT = 1e-12
_eps = EPS_DEFAULT


def set_eps(value: float) -> None:
    """Set the global default epsilon used by package functions."""
    global _eps
    _eps = float(value)


def get_eps() -> float:
    """Return the current global default epsilon value."""
    return _eps
