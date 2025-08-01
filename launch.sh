#!/bin/bash

for file in ./build/test/data/*.{osil,nl}; do
    if [[ -f "$file" ]]; then
        ./build/SHOT "$file"
    fi
done