#!bin/bash


#python 4_generate_images_orig.py --refined_neurons -o=generated_images_orig_blocked_v1_4_1 --result_file results/memorization_statistics_v1_4_1.csv --num_samples 5
python 4_generate_images_orig.py --version stabilityai/stable-diffusion-2-1 --original_images -o=generated_images_orig_unblocked_v2_1_additional_prompts --result_file prompts/additional_laion_prompts.csv --num_samples 1

