#!/bin/bash

TEMPLATE_PATH=$1
TEMPLATE_PATH="$(realpath "$TEMPLATE_PATH")"
root_path=$(pwd)

relative_path() {
    local target="$1"
    local source="${2:-$(pwd)}"
    python3 -c "import os.path; print(os.path.relpath('$target', '$source'))"
}

find . -type f -name "*.md" | while read -r markdown_file; do
    file_dir=$(dirname "$markdown_file")
    base_name=$(basename "$markdown_file" .md)

    relroot=$(relative_path "$root_path" "$file_dir")

    (cd "$file_dir" && \
    pandoc -f markdown "$base_name.md" --template="$TEMPLATE_PATH"\
        -o "$base_name.html" --mathml --citeproc --csl "$relroot/ieee.csl"\
        --metadata csspath="$relroot"\
        --metadata year=$(date +%Y)\
        --metadata author="the internet")
done
