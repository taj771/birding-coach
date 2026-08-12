#!/usr/bin/env bash
# Render OVERVIEW.md to a shareable PDF.
#
#     ./render_pdf.sh
#
# The metadata is passed on the command line rather than as YAML frontmatter in
# OVERVIEW.md, because GitHub renders frontmatter as a stray table at the top of
# the page and the markdown is read there far more often than the PDF is.
#
# monofont matters: the default LuaTeX monospace font has no box-drawing glyphs,
# so the pipeline diagram in section 4 silently loses every ─ │ └ ▼ and collapses
# into floating fragments. DejaVu Sans Mono ships with TeX Live and has them.
set -euo pipefail

quarto render OVERVIEW.md --to pdf \
  --metadata title="Birding Coach — an overview" \
  --metadata subtitle="Forecasting when to go birding, not just where" \
  --metadata author="Tharaka Jayalath" \
  --metadata date="$(date +%Y-%m-%d)" \
  --metadata toc=true \
  --metadata "geometry:margin=1in" \
  --metadata colorlinks=true \
  --metadata fontsize=10pt \
  --metadata monofont="DejaVu Sans Mono" \
  --metadata monofontoptions="Scale=0.82" \
  -o OVERVIEW.pdf

echo "wrote OVERVIEW.pdf ($(du -h OVERVIEW.pdf | cut -f1))"
