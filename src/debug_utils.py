"""Small debug-print helper used to keep normal experiment output quiet."""

import os


def is_debug_enabled():
    """
    Return whether diagnostic printing is enabled for the current process.

    The flag is propagated through the environment so Ray client actors inherit
    it when they are started by the builder.
    """
    return os.environ.get("FEDFAIRLAB_DEBUG", "0") == "1"


def debug_print(*args, **kwargs):
    """
    Print only when FedFairLAB debug mode is enabled.

    Args:
        *args: Positional arguments forwarded to ``print``.
        **kwargs: Keyword arguments forwarded to ``print``.
    """
    if is_debug_enabled():
        print(*args, **kwargs)
