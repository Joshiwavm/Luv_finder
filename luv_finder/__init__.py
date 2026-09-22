"""Blind spectral-line finding in the UV plane for interferometric data."""

from .data import DataHandler
from .matchedfilter import MatchedFilter
from .model import Gaussian, Model

__all__ = ["DataHandler", "Gaussian", "MatchedFilter", "Model", "MockObservation"]


def __getattr__(name):
    # MockObservation needs casatasks; import lazily so the core package works without CASA.
    if name == "MockObservation":
        from .mock import MockObservation

        return MockObservation
    raise AttributeError(name)
