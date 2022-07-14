#!/bin/bash

view_invariant1=(1 0)
number_dim=(2 3)
split_side=(0 1)
grid_search=(0 1)


# Iterate the string array using for loop
for v1 in ${view_invariant1[@]}; do
   for nd in ${number_dim[*]}; do
       for gs in ${grid_search[*]}; do
           for si in ${split_side[*]}; do
               echo "python main.py -v1 $v1 -nd $nd -g $gs -si $si"
               python main.py -d -v1 $v1 -nd $nd -g $gs -si $si
           done
       done
   done
done
