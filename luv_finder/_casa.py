"""Lazy CASA imports with the session log kept out of the working directory.

``casatools`` and ``casatasks`` each write a ``casa-<timestamp>.log`` into the
current directory the moment they are imported, which quickly litters the repo.
``casaconfig`` decides that path at import time, so it has to be set *before*
the first CASA import -- hence every CASA-touching module in this package goes
through :func:`tools` / :func:`tasks` rather than importing CASA directly.

Set ``LUV_CASA_LOG_DIR`` to override the destination (default: ``logs/``).
"""

from __future__ import annotations

import os
from pathlib import Path

_configured = False


def log_dir() -> Path:
    """Directory CASA logs are written to (created on demand)."""
    d = Path(os.environ.get("LUV_CASA_LOG_DIR", "logs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def configure_logging() -> None:
    """Point casaconfig at ``<log_dir>/casa.log``; a no-op after the first call.

    Call this before importing ``casatools``/``casatasks`` by any route other
    than :func:`tools` / :func:`tasks`: casaconfig fixes the log destination at
    the first CASA import and it cannot be changed afterwards.
    """
    global _configured
    if _configured:
        return
    _configured = True
    try:
        from casaconfig import config
    except ImportError:  # CASA not installed; the NPZ path does not need it
        return
    config.logfile = str(log_dir() / "casa.log")


def tools():
    """The ``casatools`` module, with logging configured."""
    configure_logging()
    import casatools

    return casatools


def tasks():
    """The ``casatasks`` module, with logging configured."""
    configure_logging()
    import casatasks

    return casatasks
