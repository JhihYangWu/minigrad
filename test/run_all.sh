#!/bin/bash

test_scripts=$(ls test_*.py)

num_passed=0
total_num=0
for file in ${test_scripts};
do
    python3 ${file} &> /dev/null
    exit_code=$?
    if test "${exit_code}" -eq "0"
    then
        echo "Passed ${file}"
        num_passed=$((num_passed+1))
    else
        echo "Failed ${file}"
    fi
    total_num=$((total_num+1))
done

echo "${num_passed}/${total_num} tests passed."

