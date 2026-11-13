"""Sphinx configuration for the pennylane-maestro documentation.

Build from the repository root:

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs docs/_build/html

The API reference imports ``pennylane_maestro``, so the package must be installed.
"""

from importlib.metadata import version as installed_version

project = "pennylane-maestro"
author = "Qoro Quantum"
copyright = "2026, Qoro Quantum"
release = installed_version("pennylane-maestro")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

exclude_patterns = ["_build"]

myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# API reference. Docstrings are Google style. api.rst lists the members to show,
# leaving out the ones every PennyLane device has.
autodoc_default_options = {"member-order": "bysource"}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pennylane": ("https://docs.pennylane.ai/en/stable", None),
    "maestro": ("https://qoroquantum.github.io/maestro", None),
}

# Theme. Matches the Maestro and Divi docs: Furo with Qoro's brand colours and fonts.
html_theme = "furo"
html_title = f"{project} {release}"
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "navigation_with_keys": True,
    "source_repository": "https://github.com/QoroQuantum/pennylane-maestro/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#CC3366",
        "color-brand-content": "#CC3366",
        "font-stack": "Poppins, sans-serif",
        "font-stack--headings": "Poppins, sans-serif",
        "font-stack--monospace": "ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, Consolas, 'Liberation Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FDB9FD",
        "color-brand-content": "#FDB9FD",
        "color-background-primary": "#131313",
        "color-background-secondary": "#1a1a1a",
        "color-foreground-primary": "#E9EEF6",
    },
}
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# GitHub renders README anchors client-side, so linkcheck can't see them.
linkcheck_anchors_ignore_for_url = [r"https://github\.com/"]
