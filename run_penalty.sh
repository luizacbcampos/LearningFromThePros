#!/bin/bash

view_invariant2=(1 0)
penalty_location=(0 1)
hand=(0 1)
height=(0 1)

# Iterate the string array using for loop
for v2 in ${view_invariant2[@]}; do
   for p in ${penalty_location[*]}; do
       for h in ${hand[*]}; do
           for hei in ${height[*]}; do
               echo "python main.py -v2 $v2 -p $p -ha $h -he $hei"
               python main.py -d -v2 $v2 -p $p -ha $h -he $hei
           done
       done
   done
done
