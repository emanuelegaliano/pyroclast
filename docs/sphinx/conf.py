import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

project = "pyroclast"
author = "emanuelegaliano"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       # NumPy and Google docstring style
    "sphinx.ext.viewcode",       # links to source code
    "sphinx.ext.intersphinx",    # links to numpy/python docs
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "alabaster"
