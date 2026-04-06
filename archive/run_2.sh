#!/bin/bash

# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=72:15:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j.out # File to which STDERR will be written

# email notifications: Get email when your job starts, stops, fails, completes...
# Set email address
#SBATCH --mail-user=jimmy.di@uwaterloo.ca
# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=ALL
 
# Load up your conda environment
# Set up environment on snorlax-login.cs or in interactive session (use `source` keyword instead of `conda`)

source activate nemo

for i in $(seq 0 11);
do
    #python3.11 -u 3_detect_memorized_neurons.py -d prompts/memorized_laion_prompts_cluster_"$i"_embeddings.csv -o results/memorization_statistics_r_cluster_"$i"_embeddings_0415.csv
    python3.11 4_generate_images.py --refined_neurons -o=generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_0415_50_1 --result_file results/memorization_statistics_r_cluster_"$i"_embeddings_0415_v1_4.csv --num_samples 50 -s 1

done