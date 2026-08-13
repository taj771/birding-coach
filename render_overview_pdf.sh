#!/usr/bin/env bash
# Render overview.html to a shareable PDF via headless Chrome.
#
#     ./render_overview_pdf.sh
#
# overview.html is the source of the published web version, which is hosted
# with a <!doctype>, <head> and <body> wrapped around it. That wrapper is
# added here too, so the file rendered locally is the same file that is
# published rather than a second copy that drifts from it.
#
# Chrome rather than Quarto because this page is hand-written CSS — grids,
# custom properties, a print stylesheet. Quarto would rebuild it from markdown
# and throw all of that away. OVERVIEW.md keeps its own renderer in
# render_pdf.sh; the two documents are deliberately different objects.
set -euo pipefail

cd "$(dirname "$0")"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || CHROME="$(command -v chromium)" || {
  echo "No Chrome or Chromium found."
  echo "Open overview.html in a browser and use File > Print > Save as PDF."
  exit 1
}

WRAPPED="$(mktemp -d)/overview.html"
{
  echo '<!doctype html><html lang="en"><head><meta charset="utf-8">'
  echo '<meta name="viewport" content="width=device-width,initial-scale=1">'
  echo '</head><body>'
  cat overview.html
  echo '</body></html>'
} > "$WRAPPED"

# --no-pdf-header-footer suppresses Chrome's own date and file:// URL, which
# otherwise print across the top of every page and look like a draft.
"$CHROME" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf-no-header \
  --print-to-pdf="$PWD/overview.pdf" \
  "file://$WRAPPED" 2>/dev/null

rm -rf "$(dirname "$WRAPPED")"
echo "wrote overview.pdf ($(du -h overview.pdf | cut -f1))"
