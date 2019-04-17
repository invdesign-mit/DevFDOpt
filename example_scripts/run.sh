#!/bin/bash

#Job=0 to print epsilon and/or E fields
#Job=1 to optimize
#Job anything else to test gradients

np=${1}
optionsfile=${2}

mpirun -np ${np} ./farfieldoptsym_exec \
       -options_file ${optionsfile} \
       -amp_real 0.123,0.137,0.120,1,1 \
       -log_summary log.txt \
       -init_dummy_var 0

