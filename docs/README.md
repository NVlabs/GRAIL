# Documentation

Source for the GRAIL documentation website.

## Building Locally

```bash
pip install -r docs/requirements.txt
cd docs
make html
```

The built site is at `build/html/index.html`.

## Live Preview

```bash
cd docs/build/html
python -m http.server 8000
# → http://localhost:8000
```

## Clean Build

```bash
cd docs
make clean && make html
```

## Deployment

Pushes to the `main` branch trigger `.github/workflows/docs.yml`,
which builds with Sphinx and publishes to GitHub Pages:
**https://nvlabs.github.io/GRAIL/**

## Layout

- `source/conf.py` — Sphinx config (sphinx-book-theme, NVIDIA branding)
- `source/index.rst` — landing page / toctree
- `source/*.md` — content pages (MyST markdown)
- `source/_static/css/custom.css` — NVIDIA-green theme overrides

Style follows the
[GR00T-WholeBodyControl docs](https://nvlabs.github.io/GR00T-WholeBodyControl/).

## Verify links from the published site

Keep the HTML built from the currently published revision before rewriting a guide. From the repository root, run the following after the strict HTML build. It checks links from the old pages, including navigation entries and section permalinks, against the proposed site. Also follow links that promise a command and confirm the destination still shows that command.

```bash
python3 docs/check_published_links.py /path/to/published/html docs/build/html --readme /path/to/published/README.md --proposed-readme README.md
```

For a rewrite, inventory the sections and runnable examples in the previously published site. Record where each command, prerequisite, output and recovery step can be found. Restore useful procedures against the current CLI and explain any retired interface. Link checks alone cannot detect a working command removed from a surviving page. Use the paper’s terms for the tracking policies and keep implementation identifiers in code formatting.
