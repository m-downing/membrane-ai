#!/bin/bash
# Fetch the LongMemEval-S cleaned dataset from Hugging Face.
# Run this once after cloning to enable benchmark runs.

set -e
mkdir -p benchmarks/data
cd benchmarks/data

if [ -f "longmemeval_s_cleaned.json" ]; then
    echo "Dataset already present at benchmarks/data/longmemeval_s_cleaned.json"
    exit 0
fi

echo "Downloading LongMemEval-S (cleaned) from Hugging Face..."
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

echo "Done."
echo "Verify: python3 -c \"import json; print(f'{len(json.load(open(\"longmemeval_s_cleaned.json\")))} items loaded')\""
