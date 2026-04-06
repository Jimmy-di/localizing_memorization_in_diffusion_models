
#!bin/bash
for i in $(seq 0 11);
do
    python load_n_graph.py -f results3/test_unblocked_"$i".png -c "$i" -u
    python load_n_graph.py -f results3/test_blocked_"$i".png -c "$i" 
done
