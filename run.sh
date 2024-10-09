#!bin/bash
echo "Threshold : $1"
echo "Noise Coeff : $2"
echo "Noise Coeff 2: $3"
echo "Noise Coeff 3: $4"
echo "Noise Coeff 4: $5"
echo "Noise Coeff 5: $6"
echo "Noise Coeff 6: $7"

python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$2.csv --pairwise_ssim_threshold $1 #--noise_mu $2 
python 4_generate_images.py --original_images -o=generated_images_unblocked_r_$1_$2 --result_file results/memorization_statistics_r_$1_$2_v1_4.csv --num_samples 5
python 4_generate_images.py --refined_neurons -o=generated_images_blocked_r_$1_$2 --result_file results/memorization_statistics_r_$1_$2_v1_4.csv --num_samples 5



# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$3.csv --pairwise_ssim_threshold $1 --noise_mu $3 
# python 4_generate_images.py --original_images -o=generated_images_unblocked_r_$1_$3 --result_file results/memorization_statistics_r_$1_$3_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_blocked_r_$1_$3 --result_file results/memorization_statistics_r_$1_$3_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$4.csv --pairwise_ssim_threshold $1 --noise_mu $4 
# python 4_generate_images.py --original_images -o=generated_images_unblocked_r_$1_$4 --result_file results/memorization_statistics_r_$1_$4_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_blocked_r_$1_$4 --result_file results/memorization_statistics_r_$1_$4_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$5.csv --pairwise_ssim_threshold $1 --noise_mu $5 

# python 4_generate_images.py --original_images -o=generated_images_unblocked_r_$1_$5 --result_file results/memorization_statistics_r_$1_$5_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_blocked_r_$1_$5 --result_file results/memorization_statistics_r_$1_$5_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$6.csv --pairwise_ssim_threshold $1 --noise_mu $6 

# python 4_generate_images.py --original_images -o=generated_images_unblocked_r_$1_$6 --result_file results/memorization_statistics_r_$1_$6_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_blocked_r_$1_$6 --result_file results/memorization_statistics_r_$1_$6_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$7.csv --pairwise_ssim_threshold $1 --noise_mu $7 

# python 4_generate_images.py --original_images -o=generated_images_unblocked_r_$1_$7 --result_file results/memorization_statistics_r_$1_$7_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_blocked_r_$1_$7 --result_file results/memorization_statistics_r_$1_$7_v1_4.csv --num_samples 5

# python -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_20.csv -o results/memorization_statistics_r_$1_$8.csv --pairwise_ssim_threshold $1 --noise_mu $8 



# python 4_generate_images.py --original_images -o=generated_images_unblocked_r_$1_$8 --result_file results/memorization_statistics_r_$1_$8_v1_4.csv --num_samples 5
# python 4_generate_images.py --refined_neurons -o=generated_images_blocked_r_$1_$8 --result_file results/memorization_statistics_r_$1_$8_v1_4.csv --num_samples 5
