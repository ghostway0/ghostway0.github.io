#!/bin/bash

TEMPLATE_PATH=$1
TEMPLATE_PATH="$(realpath "$TEMPLATE_PATH")"
root_path=$(pwd)

find . -type f -name "*.md" | while read -r markdown_file; do
    file_dir=$(dirname "$markdown_file")
    base_name=$(basename "$markdown_file" .md)

    (cd "$file_dir" && \
    pandoc -f markdown "$base_name.md" --template="$TEMPLATE_PATH" -o "$base_name.html" --mathml --citeproc --metadata csspath="$root_path" --csl "$root_path/ieee.csl")
done
