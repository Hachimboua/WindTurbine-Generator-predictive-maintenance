# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os;
import sys;
sys.setrecursionlimit(5000)
sys.path.insert(0, os.path.abspath('..'))
# -- Project information -----------------------------------------------------
project = 'Controlit'
copyright = '2025, Controlit'
author = 'Hachimboua BaquaAbdellah'

# The full version, including alpha/beta/rc tags
release = '1.0.0'
version = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'  # For Google-style docstrings
]

# Source file handling
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# List of patterns to ignore
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'venv',
    'README.md'
]

# -- Internationalization configuration ----------------------------------------
language = 'en'
locale_dirs = ['locale/']  # For translations if needed

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Recommended alternative to alabaster
html_static_path = ['_static']
html_logo = '_static/logo.png'  # If you have a logo
html_favicon = '_static/favicon.ico'

# Theme-specific options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# -- Extension settings -----------------------------------------------------
# MyST Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_heading_anchors = 3

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}