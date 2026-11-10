"""Small helpers used for printouts."""


def human_readable_numbers(x: int) -> str:
    """Format a count with a K/M/B suffix and one decimal, e.g. 1.2M.

    Used for parameter and token counts in startup printouts. Values below
    1000 are returned unchanged as a string.
    """
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    elif x >= 1e6:
        return f"{x/1e6:.1f}M"
    elif x >= 1e3:
        return f"{x/1e3:.1f}K"
    else:
        return str(x)
