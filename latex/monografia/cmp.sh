#!/bin/bash

# pdflatex --shell-escape monografia.tex; bibtex monografia.tex; pdflatex --shell-escape monografia.tex; pdflatex --shell-escape monografia.tex
pdflatex monografia.tex; bibtex monografia.tex; pdflatex monografia.tex; pdflatex monografia.tex
