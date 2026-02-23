# Paper Workspace

This folder contains the manuscript and paper-specific assets.

## Files
- `paper.tex` - main manuscript
- `plots/` - figures referenced by `paper.tex`
- `figures/` - Mermaid source diagrams
- `render_mermaid_figures.py` - renders Mermaid sources into `plots/`
- `generate_results_figures.py` - regenerates results figures from frozen summary stats

## Build
Run from this folder:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

## Refresh Mermaid diagrams
Run from repo root or from this folder:

```bash
python paper/render_mermaid_figures.py
```

## Refresh results figures
Run from repo root:

```bash
python paper/generate_results_figures.py
```
