#!/usr/bin/env bash
# Render assets/architecture.tex -> assets/architecture.png (and .svg).
# Needs pdflatex (TikZ, standalone) and poppler's pdftocairo.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error architecture.tex > /dev/null
pdftocairo -png -r 220 -singlefile architecture.pdf architecture
pdftocairo -svg architecture.pdf architecture.svg
rm -f architecture.aux architecture.log
echo "wrote $(pwd)/architecture.png and architecture.svg"
