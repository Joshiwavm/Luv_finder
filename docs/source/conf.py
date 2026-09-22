"""Sphinx configuration (furo theme, notebook tutorials rendered but not executed)."""

from __future__ import annotations

project = "Luv_finder"
copyright = "2026, Joshiwa van Marrewijk"
author = "Joshiwa van Marrewijk"
html_title = "Luv_finder"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "**.ipynb_checkpoints"]
html_theme = "furo"
html_static_path = ["_static"]
pygments_style = "sphinx"
pygments_dark_style = "monokai"

autosummary_generate = True
numpydoc_show_class_members = False
autodoc_mock_imports = ["casatools", "casatasks", "casadata"]
nbsphinx_execute = "never"  # tutorials need CASA + mock data; commit them executed

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
