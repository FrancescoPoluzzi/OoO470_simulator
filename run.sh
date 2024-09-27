#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh </path/to/input.json> </path/to/output.json>"
    exit 1
fi

# Assign command line arguments to variables
INPUT_FILE=$1
OUTPUT_FILE=$2

# Run the Python simulator with the input and output file arguments
python simulator.py "$INPUT_FILE" "$OUTPUT_FILE"
