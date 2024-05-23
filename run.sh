#!/bin/bash

echo "Run Hand In 4"

if [ ! -d "data" ]; then
    echo "Create data directory"
    mkdir data
fi

if [ ! -e "data/galaxy_data.txt" ]; then
    echo "Download galaxy_data.txt"
    wget -O data/galaxy_data.txt https://home.strw.leidenuniv.nl/~daalen/Handin_files/galaxy_data.txt
fi

if [ ! -e "fig1a.png" ]; then
    echo "Run ex1.py"
    python3 ex1.py
fi

if [ ! -e "fig2a.png" ]; then
    echo "Run ex2.py"
    python3 ex2.py
fi

if [ ! -e "fig3a.png" ]; then
    echo "Run ex3.py"
    python3 ex3.py
fi

echo "Generating the pdf"
pdflatex NUR-4.tex > latex_output1.txt
#bibtex template.aux > bibtex_output.txt
pdflatex NUR-4.tex > latex_output2.txt
pdflatex NUR-4.tex > latex_output3.txt
