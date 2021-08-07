#!/bin/bash

for c in bags_and_cases bluetooth boots keyboards tv vacuums; do
    upper=`echo "$c" | tr '[:lower:]' '[:upper:]'`
    python extract_data.py --source ./data/preprocessed/"$upper"_MATE.hdf5 \
                           --output ./data/"$c"_train.json
    python extract_data.py --source ./data/preprocessed/"$upper"_MATE_TEST.hdf5 \
                           --output ./data/"$c"_test.json
done