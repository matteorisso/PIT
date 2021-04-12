#!/bin/bash

# Before running always check that regularization_strength and 
# gamma_threshold are equal respectively to 1e-4 and 5e-1 in config.py

#reg_strength=(0 0.005 0.01 0.02 0.04 0.1)
#reg_strength=(0.1 0.04 0.02 0.01 0.005 0.)
reg_strength=(5e-4 1e-5 5e-5 1e-6)
#threshold=(1e-2 2.5e-2 5e-2 7.5e-2 1e-1)

old_reg="reg_strength = 1e-6"
#old_th="gamma_threshold = 2.5e-2"

for i in "${reg_strength[@]}"
do
    #new_reg="reg_strength = $i"
    #echo old_reg : $old_reg
    #echo new_reg : $new_reg
    #sed -i "s/$old_reg/$new_reg/" config.py
    #for j in "${threshold[@]}"
    #do
    #    new_th="gamma_threshold = $j"
    #    echo old_th : $old_th
    #    echo new_th : $new_th
    #    sed -i "s/$old_th/$new_th/" config.py

    python3 pit.py PPG_Dalia $i 20

        #old_th=$new_th
    #done
    #old_reg=$new_reg
done
