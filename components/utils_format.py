# components/utils_format.py


def fmt_int(n: int | float | None) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return "—"

def fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}%"

def fmt_date(dt) -> str:
    try:
        return str(dt.date())
    except Exception:
        return "—"
    

def fmt_compact(x: int | float | None) -> str:
    """
    Format numbers into compact form:
    - 1,234   -> 1.23K
    - 56,789  -> 56.79K
    - 1,234,567 -> 1.23M
    """
    if x is None:
        return "—"
    try:
        x = float(x)
    except (ValueError, TypeError):
        return "—"

    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    elif abs(x) >= 1_000:
        return f"{x/1_000:.2f}K"
    else:
        return f"{x:.0f}"