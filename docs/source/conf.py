# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

project = 'Mobius'
copyright = '2023, Eberhardt J., Lill M., Schwede T.'
author = 'Eberhardt J., Lill M., Schwede T.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'numpydoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.imgconverter']

templates_path = ['_templates']
exclude_patterns = []

html_show_copyright = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Activate autosectionlabel plugin
autosectionlabel_prefix_document = True
autodoc_member_order = 'bysource'

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
