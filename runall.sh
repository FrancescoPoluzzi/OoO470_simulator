#!/bin/bash

# ./build.sh

# for tnum in ./given_tests/*
# do
#     ./run.sh ${tnum}/input.json ${tnum}/user_output.json
# done

./build.sh

for dir in ./given_tests/*; do
    if [ -d "$dir" ]; then  # Check if it's a directory
        input="${dir}/input.json"
        output="${dir}/user_output.json"  # Specify the output filename
        ./singlerun.sh "$input" "$output"
    fi
done
