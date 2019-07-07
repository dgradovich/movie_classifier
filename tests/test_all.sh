#!/bin/sh

cd tests/unit/
for test in test_*.py
do
    python3 "${test}" -v
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo "failed $CMD"
        exit 1
    fi
done

cd ../..

RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "failed $CMD"
    exit 1
fi