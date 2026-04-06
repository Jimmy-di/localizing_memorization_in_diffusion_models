#!bin/bash

for i in $(seq 0 11);
do
    pyiqa qalign -t generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_0428  --device cuda

done