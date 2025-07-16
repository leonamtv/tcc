#!/bin/bash

pdflatex slides.tex; bibtex slides.tex; pdflatex slides.tex; pdflatex slides.tex
