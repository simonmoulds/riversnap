import os
import sys

# Add src to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

project = "riversnap"
author = "riversnap contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
]
nbspinx_execute = "never"

autosummary_generate = True

autodoc_default_options = {
    "members": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
