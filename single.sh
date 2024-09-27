#!/bin/bash
./run.sh ./given_tests/"$1"/input.json ./given_tests/"$1"/user_output.json
python ./compare.py ./given_tests/"$1"/user_output.json -r ./given_tests/$1/output.json