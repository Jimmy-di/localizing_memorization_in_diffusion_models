#!bin/bash
echo "Threshold : $1"

python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_reduced.csv -o results/memorization_statistics_r_$1.csv --pairwise_ssim_threshold $1 > step3_r_$1.txt
python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_reduced_modified.csv -o results/memorization_statistics_rm_$1.csv --pairwise_ssim_threshold $1 > step3_rm_$1.txt

python 4_generate_images.py --original_images -o=generated_images_unblocked_r_$1 --result_file results/memorization_statistics_r_$1_v1_4.csv
python 4_generate_images.py --refined_neurons -o=generated_images_blocked_r_$1 --result_file results/memorization_statistics_r_$1_v1_4.csv

python 4_generate_images.py --original_images -o=generated_images_unblocked_rm_$1 --result_file results/memorization_statistics_rm_$1_v1_4.csv
python 4_generate_images.py --refined_neurons -o=generated_images_blocked_rm_$1 --result_file results/memorization_statistics_rm_$1_v1_4.csv
