# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Predicting the Unpredictable'
copyright = '2025, Team Sally'
author = ('Fuchs, Ava R; HU, Wenzhi; LI, Zhifeng; Pejic, Milica; '
          'RAO, Tingyu; Takatani, Masahiko; WU, Jiayu; Yu, Zirui; ZHANG, Wanyu')
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__, __call__',
    'undoc-members': True,
    'show-inheritance': True,
}

# autodoc_typehints = 'description'
autosummary_generate = True

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

# Create _static directory if it doesn't exist (for custom CSS/JS)
os.makedirs(os.path.join(os.path.dirname(__file__), '_static'), exist_ok=True)
