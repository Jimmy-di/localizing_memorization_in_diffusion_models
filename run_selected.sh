#!bin/bash
echo "Threshold : $1"


#python -u 3_detect_memorized_neurons.py -d prompts/memorized_prompts_few.csv -o results/memorization_statistics_few_2_$1.csv

python 4_generate_images.py --original_images -o=generated_images_unblocked_few_$1_2 --result_file results/memorization_statistics_few_2_$1_v1_4.csv --num_samples 5
python 4_generate_images.py --refined_neurons -o=generated_images_blocked_few_$1_2 --result_file results/memorization_statistics_few_2_$1_v1_4.csv --num_samples 5