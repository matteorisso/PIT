#!/bin/bash

reg_strength=(5e-4 1e-5 5e-5 1e-6)
warmup=20

for i in "${reg_strength[@]}"
do

    python3 pit.py PPG_Dalia $i $warmup
    
done
