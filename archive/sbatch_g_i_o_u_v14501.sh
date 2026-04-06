#!/bin/bash

# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=372:15:00
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
# Set up environment on snorlax-login.cs or in interactive session (use `source` keyword instead of `conda`)W
source nemo3/bin/activate
# Task to run
python3 7_bg_score_single.py
# python3 4_generate_images_orig.py --original_images -o=generated_images_unblocked_v1_4_50_1 --result_file results/memorization_statistics_v1_4_1.csv --num_samples 50
