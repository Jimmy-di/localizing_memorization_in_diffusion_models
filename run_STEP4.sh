#!bin/bash
echo "Threshold : $1"
#echo "Noise Coeff : $2"
#echo "Noise Coeff 2: $3"
#echo "Noise Coeff 3: $4"
#echo "Noise Coeff 4: $5"
#echo "Noise Coeff 5: $6"
#echo "Noise Coeff 6: $7"

python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.0
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.1
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.2
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.3
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.4
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.5
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.6
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.7
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.8
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 0.9
python -u 4_compare_latents.py --prompt results/memorization_statistics_r_v1_4.csv --scaling_factor 1.0