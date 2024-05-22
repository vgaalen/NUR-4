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