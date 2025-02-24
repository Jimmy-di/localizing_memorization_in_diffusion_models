#!bin/bash
echo "Threshold : $1"

for i in $(seq 0 3);
do
    #python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_cluster_"$i"_embeddings.csv -o results/memorization_statistics_r_cluster_"$i"_embeddings_$1.csv --pairwise_ssim_threshold $1
    #python 4_generate_images_orig.py --original_images -o=generated_images_orig_unblocked_r_cluster_"$i"_embeddings_block_all_g1_$1 --result_file results/memorization_statistics_r_cluster_"$i"_embeddings_$1_v1_4.csv --num_samples 5
    python 4_generate_images_orig.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_g1_$1 --result_file results/memorization_statistics_r_cluster_"$i"_embeddings_$1_v1_4.csv --num_samples 5 -g 1
    python 4_generate_images_orig.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_g0.1_$1 --result_file results/memorization_statistics_r_cluster_"$i"_embeddings_$1_v1_4.csv --num_samples 5 -g 0.1
    #python 4_generate_images_orig.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_g5_$1 --result_file results/memorization_statistics_r_cluster_"$i"_embeddings_$1_v1_4.csv --num_samples 5 -g 5
    #python 4_generate_images_orig.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_g4_$1 --result_file results/memorization_statistics_r_cluster_"$i"_embeddings_$1_v1_4.csv --num_samples 5 -g 4
    #python 4_generate_images_orig.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_g9_$1 --result_file results/memorization_statistics_r_cluster_"$i"_embeddings_$1_v1_4.csv --num_samples 5 -g 9

done
#python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_cluster_all_embeddings.csv -o results/memorization_statistics_r_cluster_all_embeddings_$1.csv --pairwise_ssim_threshold $1
#python 4_generate_images_orig.py --original_images -o=generated_images_orig_unblocked_r_cluster_all_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_all_embeddings_$1_v1_4.csv --num_samples 3
#python 4_generate_images_orig.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_all_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_all_embeddings_$1_v1_4.csv --num_samples 3


# python -u 3_detect_memorized_neurons.py -d pmemorization_statistics_r_v1_4.csv 
# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_cluster_0_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_0_embeddings_$1_v1_4.csv --num_samples 1
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_0_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_0_embeddings_$1_v1_4.csv --num_samples 1

# #python -u 3_detect_memorized_neurons.py -d memorized_laion_prompts_cluster_4_embeddings.csv -o results/memorization_statistics_r_cluster_4_embeddings_$1.csv --pairwise_ssim_threshold $1
# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_cluster_1_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_1_embeddings_$1_v1_4.csv --num_samples 1
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_1_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_1_embeddings_$1_v1_4.csv --num_samples 1


# #python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_cluster_9_embeddings.csv -o results/memorization_statistics_r_cluster_9_embeddings_$1.csv --pairwise_ssim_threshold $1
# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_cluster_2_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_2_embeddings_$1_v1_4_1.csv --num_samples 1
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_2_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_2_embeddings_$1_v1_4_1.csv --num_samples 1

# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_cluster_3_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_3_embeddings_$1_v1_4_1.csv --num_samples 1
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_3_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_3_embeddings_$1_v1_4_1.csv --num_samples 1

# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_cluster_6_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_6_embeddings_$1_v1_4.csv --num_samples 1
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_6_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_6_embeddings_$1_v1_4.csv --num_samples 1

# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_cluster_5_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_5_embeddings_$1_v1_4.csv --num_samples 1
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_5_embeddings_block_all_$1 --result_file results/memorization_statistics_r_cluster_5_embeddings_$1_v1_4.csv --num_samples 1

#python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_cluster_0_block_all --result_file results/memorization_statistics_r_cluster_0_v1_4.csv --num_samples 5
#python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_0_block_all0_5 --result_file results/memorization_statistics_r_cluster_0_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$4.csv --pairwise_ssim_threshold $1 --noise_mu $4 
# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_$1_$4 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_$1_$4 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$5.csv --pairwise_ssim_threshold $1 --noise_mu $5 

# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_$1_$5 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_$1_$5 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$6.csv --pairwise_ssim_threshold $1 --noise_mu $6 

# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_$1_$6 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_$1_$6 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$7.csv --pairwise_ssim_threshold $1 --noise_mu $7 

# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_$1_$7 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_$1_$7 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$8.csv --pairwise_ssim_threshold $1 --noise_mu $8 

# python 4_generate_images.py --original_images -o=generated_images_orig_unblocked_r_$1_$8 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_$1_$8 --result_file results/memorization_statistics_r_$1_0_v1_4.csv --num_samples 5
