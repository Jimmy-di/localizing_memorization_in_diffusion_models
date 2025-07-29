#!bin/bash

#python -u 3_detect_memorized_neurons.py -d prompts/memorized_prompts_few.csv -o results/memorization_statistics_few_2_$1.csv

python 4_generate_images.py --original_images -o=generated_images_unblocked_v1_4_50 --result_file results/memorization_statistics_v1_4_1.csv --num_samples 50
