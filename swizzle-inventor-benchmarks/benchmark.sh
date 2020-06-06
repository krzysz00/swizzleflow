#!/usr/bin/env zsh
for file in "$@"; do
    local filename=$(basename "$file")
    local typ=$(echo $filename | cut -d_ -f1)
    local spec_name=$(echo $filename | cut -d_ -f2- | sed -e 's/.rkt$//' -e 's#_#/#')
    local reruns=1
    if [[ $typ == "small" ]]; then
        reruns=3
    fi
    for i in {0..$(($reruns - 1))}; do
        echo $filename - $i
        racket $file | sed -E -e "s#.*real time: ([0-9]+) .*#run:${spec_name} \1#"
    done
done
