# conf.py

import os
import sys
# Increase recursion limit for large/nested toctrees (essential for your RecursionError)
sys.setrecursionlimit(2000) # You can try a higher number like 3000 if 2000 isn't enough

# Add these paths to sys.path
# This ensures Python can find your 'Dashboard_App' package
sys.path.insert(0, os.path.abspath('..')) # Path to your project root (parent of docs)
sys.path.insert(0, os.path.abspath('../Dashboard_App')) # Path directly to Dashboard_App

# ... (rest of your conf.py) ...

# Ensure autodoc is enabled
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
]

# ... (rest of your conf.py) ...

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme' # Read the Docs theme

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False # Set to True if you use NumPy style
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MyST-Parser configuration (if you included README.md)
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# autosummary configuration (if you want to use it)
autosummary_generate = True

# master_doc = 'index' # For Sphinx < 2.0, use this instead of root_doc
root_doc = 'index' # For Sphinx >= 2.0