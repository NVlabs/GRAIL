# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "GRAIL"
copyright = "2026, NVIDIA"
author = "NVIDIA DAIR Team"
release = "0.1.0"
version = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

suppress_warnings = ["ref.python"]

# -- MyST Parser configuration -----------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "html_image",
]

myst_heading_anchors = 3

# Single source of truth for the public repo URL. Drives `extlinks` below,
# the theme's GitHub-backed edit/issue links, so a repo rename only touches this
# one line.
# Caveats: the project links in index.rst sit inside `raw:: html` blocks
# and need a manual edit on rename; the `git clone` URL in installation.md
# lives inside a code fence and must also be edited manually.
_REPO_URL = "https://github.com/NVlabs/GRAIL"

# Custom roles for in-repo links. In markdown:
#     {src}`grail/visualization/`              → tree-view link
#     {blob}`grail/data_export/foo.py`         → file-view link
#     {blob}`override README <path/to/file>`   → custom link text
# Replaces an earlier `{{ repo }}` MyST-substitution mechanism, which does
# not reliably work inside link URL parens (myst-parser issue #297). Roles
# work in URL contexts because Sphinx resolves them before HTML emission.
extlinks = {
    "src": (_REPO_URL + "/tree/main/%s", "%s"),
    "blob": (_REPO_URL + "/blob/main/%s", "%s"),
}

# -- Options for HTML output -------------------------------------------------

html_title = "GRAIL Documentation"
html_theme = "sphinx_book_theme"
html_show_copyright = True
html_show_sphinx = False
html_last_updated_fmt = ""

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "path_to_docs": "docs/source/",
    "collapse_navigation": True,
    "repository_url": _REPO_URL,
    "repository_branch": "main",
    "use_repository_button": False,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "use_sidenotes": True,
    "logo": {
        "text": "GRAIL Documentation",
    },
    "icon_links": [
        {
            "name": "GRAIL",
            "url": "https://nvlabs.github.io/GRAIL/",
            "icon": "fa-solid fa-globe",
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick Links",
}

language = "en"
