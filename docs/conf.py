# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import inspect  # to help us check the arguments we receive in our docstring check
import os
import re  # for regex checking
import subprocess  # to initiate sphinx-apidoc to build .md files
import sys
import warnings  # to throw warnings (not errors) for malformed docstrings

import mock  # if we need to force mock import certain libraries autodoc_mock_imports fails ons
from sphinx.builders.html import StandaloneHTMLBuilder

# our project officially 'begins' in the parent aka root project directory
# since we do not separate source from build we can simply go up one directory
sys.path.insert(0, os.path.abspath('../src'))

# if we ever have images, we'll be using the supported_image_types
# argument to set the desired formats we wish to support

# -- Project information -----------------------------------------------------

project = 'ark'
copyright = '2020, Angelo Lab'
author = 'Angelo Lab'

# whether we are on readthedocs or not
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# grab which version we want: needs to be either stable or latest
rtd_version = os.environ.get('READTHEDOCS_VERSION', 'latest')
if rtd_version not in ['stable', 'latest']:
    rtd_version = 'stable'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['IPython.sphinxext.ipython_console_highlighting',  # syntax-highligyting ipython interactive sessions
              'sphinx.ext.autodoc',  # allows you to generate documentation from docstrings (STAR)
              # allows you to refer sections aka link to them (STAR)
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.coverage',  # get coverage statistics (STAR)
              'sphinx.ext.doctest',  # provide a test code snippits
              # link to other project's documentation, needed if a cross-reference has no matching target in current documentation
              'sphinx.ext.intersphinx',
              # generates a .nojekyll file on generated HTML directory, allows publishing to GitHub pages
              'sphinx.ext.githubpages',
              'sphinx.ext.napoleon',  # support for Google style docstrings (STAR)
              'sphinx.ext.todo',  # support fo TODO
              'sphinx.ext.viewcode',  # support for adding links to highlighted source code, looks at Python object descriptions and tries to find source files where objects are contained
              'm2r2']  # allows you to include Markdown files in .rst, use mdinclude for this, choosing this over m2r because m2r is not supported anymore

# set parameter to read Google docstring and not NumPy
# redundant to add since it's default True but good to know
napoleon_google_docstring = True

# contains list of modules to be marked up
# will ensure 'clean' imports of all the following libraries
autodoc_mock_imports = ['cryptography',
                        'datasets',
                        'feather',
                        'google',
                        'h5py',
                        'ipywidgets',
                        'natsort',
                        'numpy',
                        'matplotlib',
                        'natsort',
                        'palettable',
                        'pandas',
                        'pyarrow',
                        'pyFlowSOM',
                        'skimage',
                        'sklearn',
                        'scipy',
                        'seaborn',
                        'statsmodels',
                        'spatial_lda',
                        'tables',
                        'tifffile',
                        'alpineer',
                        'umap',
                        'xarray',
                        'twisted',
                        'kiosk_client',
                        'mpl_toolkits',
                        'tqdm',
                        'ark.utils._bootstrapping',
                        'xmltodict']

# prefix each section label with the name of the document it is in, followed by a colon
# autosection_label_prefix_document = True
autosectionlabel_prefix_document = True

# what role to use for text marked up like `...`, for example this will allow you to parse `filter  ` as a cross-reference to Python function 'filter'
default_role = 'py:obj'

# use recommonmark's CommonMarkParser to parse markdown
# might be unnecessary with m2r2 but we'll see
source_parsers = {'.md': 'recommonmark.parser.CommonMarkParser'}

# which types of file extensions we want to support
# need .rst for index.rst, and also .md for Markdown
source_suffix = ['.rst', '.md']

# the path to the 'master' document, which in our case, is just index.rst
# not really needed because this is the default value, but still useful to know
master_doc = 'index'

# the language we are writing the documentation in
language = 'en'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# don't allow nbsphinx to run notebooks
nbsphinx_execute = 'never'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_rtd/landing.md', '_markdown/ark.md',
                    '_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# custom 'stuff' we want to ignore in nitpicky mode
# currently empty, I don't think we'll ever run in this
# but if we do we might consider adding
nitpick_ignore = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for Intersphinx config -------------------------------------------------

# intersphinx mapping, for when there is a cross-reference that has no matching target
# in the current documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.8', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'matplotlib': ('https://matplotlib.org/3.4.3', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None)
}

# set a maximum number of days to cache remote inventories
intersphinx_cache_limit = 0


# appends README.md to landing.md and adds proper formatting
def append_readme():
    with open(os.path.join('..', 'README.md')) as fin, open(os.path.join('_rtd', 'landing.md'), 'a') as fout:
        # flag to identify when to start writing to file, doing this because we don't want to
        # include build or covarage status nor the ark-analysis heading
        seen_heading = False

        for line in fin:
            # once we've seen an h2 heading, we can start writing to file
            # we also append an extra '#' to the line to downgrade to an h3 heading
            # so that it can display comfortably on ReadTheDocs
            if line.startswith('##'):
                seen_heading = True
                line = '#' + line

            # append line to landing.md
            if seen_heading:
                # reconfigure relative paths in README.md to be relative to _rtd/landing.md
                match = re.search(r"!\[\]\((?:(?<!\bhttp).)+\b\W*\)", line)
                if match:
                    match_str = match.string[match.start():match.end()]
                    match_path = match_str[4:-1]
                    adjusted_path = os.path.join('..', match_path)
                    line = match_str[:4] + adjusted_path + match_str[-1]

                fout.write(line)


# removes unnecessary text from the .md files created by sphinx-apidoc
def trim_rtd():
    # for some reason, sphinx-apidoc generates a .md file for the entire project, remove
    os.remove(os.path.join('_markdown', 'ark.md'))

    for md in os.listdir('_markdown'):
        with open(os.path.join('_markdown', md)) as fin:
            md_lines = fin.readlines()

        # define a list of new lines we wish to append
        new_lines = []

        # whether to skip the line or not
        write_line = True

        for line in md_lines:
            # skip writing a line, doing it this way: don't want nested if statements here
            if not write_line:
                write_line = True
                continue

            # remove "package" from the title
            if 'package' in line:
                new_lines.append(line.replace(' package', ''))
            # don't write "Submodules" or the "---..." after it
            elif 'Submodules' in line:
                write_line = False
            elif 'module' in line:
                new_lines.append(line.replace(' module', ''))
            # don't write anything after or including Module contents
            elif 'Module contents' in line:
                break
            else:
                new_lines.append(line)

        with open(os.path.join('_markdown', md), "w") as fout:
            fout.write(''.join(new_lines))


# this we'll need to build the documentation from sphinx-apidoc ourselves
def run_apidoc(_):
    # the parent directory we will begin snaking through to find documentation
    module = '../src/ark'

    # we want the documents to be markdown files
    output_ext = 'md'

    # we want to store the documentation in the _markdown folder on the ReadTheDocs server
    output_path = '_markdown'

    # the sphinx-apidoc command to run
    cmd_path = 'sphinx-apidoc'

    # should probably remove this
    if hasattr(sys, 'real_prefix'):
        cmd_path = os.path.abspath(os.path.join(sys.prefix, 'bin', 'sphinx-apidoc'))

    # append the readme to landing.md before running sphinx apidoc build
    append_readme()

    # run sphinx-apidoc to build documentation from Google docstrings
    subprocess.check_call([cmd_path, '-f', '-T', '-s', output_ext,
                           '-o', output_path, module])

    # remove extraneous text created by sphinx-apidoc
    trim_rtd()


# check for formatting errors in the docstring not caught by RTD's backend
def check_docstring_format(app, what, name, obj, options, lines):
    if what == 'function':
        argspec = inspect.getfullargspec(obj)
        argnames = \
            argspec.args \
            + ([argspec.varargs] if argspec.varargs else []) \
            + argspec.kwonlyargs

        if len(argnames) > 0:
            # I'm leaving this one out for now since we're possibly waiting on some of these
            # normally, we should indeed be throwing an exception for this case
            # because every function needs a docstring
            if len(lines) == 0:
                # raise Exception('No docstring provided for function %s' % name)
                return

            # all docstrings need a description, if we're getting into the args list immediately
            # that is a bad, bad thing
            if len(lines) > 0 and lines[0][0] == ':':
                raise Exception('No description before args list given for %s' % name)

            # the first value of lines should always contain the start of the description
            # if there is an extra space in front, that violates how a Google doctring should look
            if lines[0][0].isspace():
                raise Exception(
                    'Your description for %s should not have any preceding whitespace' % name
                )

            # handle the Args section, all args should have an associated :param and :type
            # we need to ignore *args and **kwargs because those don't get counted in the args list
            # but do get counted in the :param list
            # TODO: fix if this causes an issue in the future
            param_args = [re.match(r':param (\S*):', line).group(1) for line in lines
                          if re.match(r':param (\S*):', line)
                          and re.match(r':param (\S*):', line).group(1) != '\\*args'
                          and re.match(r':param (\S*):', line).group(1) != '\\*\\*kwargs']
            type_args = [re.match(r':param (\S*):', line).group(1) for line in lines
                         if re.match(r':param (\S*):', line)
                         and re.match(r':param (\S*):', line).group(1) != '\\*args'
                         and re.match(r':param (\S*):', line).group(1) != '\\*\\*kwargs']

            # usually this happens when the person writing the docs does not know how to tab right
            # and ReadTheDocs gets screwed over when processing so reads something
            # in an argument description as an actual argument
            if sorted(param_args) != sorted(type_args):
                raise Exception('Parameter list: %s and type list: %s do not match in %s, an Args section formatting error likely caused this' % (
                    ','.join(param_args), ','.join(type_args), name))

            # if your parameters are not the same as the arguments in the function
            # that's bad because your docstring args section needs to match up exactly
            if sorted(param_args) != sorted(argnames):
                raise Exception('Parameter list: %s does not match arglist: %s in %s, check docstring formatting' % (
                    ','.join(param_args), ','.join(argnames), name))

            # handle cases where return values are found
            # currently, I can only check if in the case of a proper Return (:return) format (improper ones are usually handled by the above cases)
            # it also contains a proper return type (:rtype)
            # this one will be harder because for one, we cannot make the assumption that all functions return something
            # second of all, the :returns and :rtype values may look OK in the list but still be formatted weird if the user screwed up tabs for example
            if any(re.match(r':returns:', line) for line in lines):
                if not any(re.match(r':rtype', line) for line in lines):
                    raise Exception('Return value was provided but no return type specified in %s'
                                    % name)
        else:
            # every function needs a docstring, this currently is OK for now because
            # only one function is like this: create_test_extraction_data in test_utils
            if len(lines) == 0:
                raise Exception('No docstring provided for function %s' % name)

            # the first value of lines should always contain the start of the description
            # if there is an extra space in front, that violates how a Google doctring should look
            if lines[0][0].isspace():
                raise Exception(
                    'Your description for %s should not have any preceding whitespace' % name
                )

            # we need to ignore *args and **kwargs because those don't get counted in the args list
            # but do get counted in the :param list
            # TODO: fix if this causes an issue in the future
            param_args = [re.match(r':param (\S*):', line).group(1) for line in lines
                          if re.match(r':param (\S*):', line)
                          and re.match(r':param (\S*):', line).group(1) != '\\*args'
                          and re.match(r':param (\S*):', line).group(1) != '\\*\\*kwargs']

            # do not allow the user to specify an args list for a function that doesn't take args
            if len(param_args) > 0:
                raise Exception(
                    'You should not have an Args list specified for argument-less function %s'
                    % name)

            # this is like the returns check for if the function does specify arguments
            if any(re.match(r':returns:', line) for line in lines):
                if not any(re.match(r':rtype', line) for line in lines):
                    raise Exception(
                        'Return value was provided but no return type specified in %s' % name)


def setup(app):
    app.connect('builder-inited', run_apidoc)  # run sphinx-apidoc
    app.connect('autodoc-process-docstring', check_docstring_format)  # run a docstring-style check
