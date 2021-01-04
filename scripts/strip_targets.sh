#!/usr/bin/bash

if ! command -v rdf2hdt &> /dev/null
then
    echo "Missing rdf2hdt"
    exit 1
fi


function is_filetype {
    if [[ "$1" == *."$2" ]]
    then
        return 1
    fi

    return 0
    }

function is_hdt() {
    is_filetype "$1" "hdt"
    return $?
    }

function is_nt() {
    is_filetype "$1" "nt"
    return $?
    }

if [[ $# -ne 2 ]];
then
    echo "USAGE: ./strip_targets.sh <graph as nt or hdt> <target predicate>"
    exit 1
fi

graph_unstripped="$1"
target_predicate="$2"

is_hdt "$graph_unstripped"
if [ $? -eq 1 ];
then
    filename=$(basename -s .hdt "$graph_unstripped")
    hdt2rdf "$graph_unstripped" "$filename"".nt"
    graph_unstripped="$filename"".nt"
fi

is_nt "$graph_unstripped"
if [ $? -ne 1 ];
then
    echo "Expecting hdt or nt file"
    exit 1
fi

filename=$(basename -s .nt "$graph_unstripped")
grep "^.*\s<$target_predicate>\s.*\s\." "$graph_unstripped" | gzip - > samples.nt.gz
sed -e "\|^.*\s<$target_predicate>\s.*\s\.|d" "$graph_unstripped" > "$filename""_stripped.nt"

rdf2hdt "$filename""_stripped.nt" "$filename""_stripped.hdt"

echo "Delete $filename_stripped.nt to clear intermediate file"

exit 0
