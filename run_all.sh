#!/bin/bash
view_invariant1=(1 0)
number_dim=(2 3)
split_side=(0 1)
grid_search=(0 1)

view_invariant2=(1 0)
penalty_location=(0 1)
hand=(0 1)
height=(0 1)


declare -a part1=()
count=0

for v1 in ${view_invariant1[@]}; do
   for nd in ${number_dim[*]}; do
       for gs in ${grid_search[*]}; do
           for si in ${split_side[*]}; do
           		# echo "python main.py -v1 $v1 -nd $nd -g $gs -si $si"
           		# python main.py -v1 $v1 -nd $nd -g $gs -si $si
               part1[$count]="python main.py -v1 $v1 -nd $nd -g $gs -si $si"
               count=$((count + 1))
           done
       done
   done
done

declare -a part2=()
count=0
for v2 in ${view_invariant2[@]}; do
   for p in ${penalty_location[*]}; do
       for h in ${hand[*]}; do
           for hei in ${height[*]}; do
               # echo "python main.py -v2 $v2 -p $p -ha $h -he $hei"
               # python main.py -v2 $v2 -p $p -ha $h -he $hei
               part2[$count]="-v2 $v2 -p $p -ha $h -he $hei"
               count=$((count + 1))
           done
       done
   done
done

for round in {0..4} do
    for i in {0..15}; do
        # echo "$i"
    	command="${part1[$i]} ${part2[$i]}"
    	echo "$command"
    	eval "$command"
    done
done


