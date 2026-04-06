#!bin/bash

echo "Noise Coeff : $1"
#echo "Noise Coeff 2: $3"
#echo "Noise Coeff 3: $4"
#echo "Noise Coeff 4: $5"
#echo "Noise Coeff 5: $6"
#echo "Noise Coeff 6: $7"

for i in $(seq 7 13);
do

    python compute_image_mse2.py -c "$i" -t $1
    python compute_image_mse.py -c "$i" -t $1

done

