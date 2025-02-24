#!bin/bash
echo "Threshold : $1"

for i in $(seq 14 20);
do
    #python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_cluster_"$i"_embeddings.csv -o results/memorization_statistics_r_cluster_"$i"_embeddings_$1.csv --pairwise_ssim_threshold $1
    python 4_generate_images_orig.py --original_images -o=generated_images_orig_unblocked_r_cluster_"$i"_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_"$i"_embeddings_$1_v1_4.csv --num_samples 3
    python 4_generate_images_orig.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_"$i"_embeddings_$1_v1_4.csv --num_samples 3

done