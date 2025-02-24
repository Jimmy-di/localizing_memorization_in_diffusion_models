
#!bin/bash
for i in $(seq 0 20);
do
    python load_n_graph_diff.py -c "$i" -f results3/test_c"$i"_diff_abs.png
done


