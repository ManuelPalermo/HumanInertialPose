# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'HumanInertialPose'
copyright = '2021, Manuel Palermo'
author = 'Manuel Palermo'

# The full version, including alpha/beta/rc tags.
from hipose import __version__
lib_version = '.'.join(__version__.split('.')[:-1])
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.napoleon',
        'sphinx.ext.doctest',
        'sphinx.ext.coverage',
        'sphinx.ext.todo',
        'autoapi.extension',
        # 'sphinx.ext.inheritance_diagram',
]

# use auto_api to generate docs
# https://github.com/readthedocs/sphinx-autoapi
autoapi_type = 'python'
autoapi_dirs = ['../../hipose']
autoapi_ignore = ["__init__.py"]
autoapi_generate_api_docs = True
autoapi_member_order = "groupwise"  # "bysource"
suppress_warnings = ["autoapi"]
autoapi_options = ['members',
                   'undoc-members',
                   'private-members',
                   'show-inheritance',
                   'show-module-summary',
                   'special-members',
                   'imported-members',
                   # 'show-inheritance-diagram'
                   ]

# include todos?
todo_include_todos = True

# The master toctree document.
master_doc = 'index'

# The suffix of source filenames.
source_suffix = '.rst'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
