#!bin/bash
echo "Threshold : $1"

for i in $(seq 0 11);
python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_rewrite_1.csv -o results/memorization_statistics_r_rewrite_1.csv --pairwise_ssim_threshold $1
python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_rewrite_2.csv -o results/memorization_statistics_r_rewrite_2.csv --pairwise_ssim_threshold $1
python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_rewrite_3.csv -o results/memorization_statistics_r_rewrite_3.csv --pairwise_ssim_threshold $1

#python 4_generate_images.py --original_images -o=generated_images_unblocked_rm_$1 --result_file results/memorization_statistics_rm_$1_v1_4.csv
#python 4_generate_images.py --refined_neurons -o=generated_images_blocked_rm_$1 --result_file results/memorization_statistics_rm_$1_v1_4.csv
