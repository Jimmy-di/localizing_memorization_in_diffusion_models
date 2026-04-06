#!bin/bash

echo $1

for i in $(seq 14 20);
do

    python compute_image_mse2.py -c "$i" -t $1
    python compute_image_mse.py -c "$i" -t $1

done
