#!/bin/bash

# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=1324:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
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


source activate mmseg
python3 6_faiss_search.py --sname generated_images_orig_blocked_v1_4_1
#python3.11 -u 7_bg_scores.py --sname generated_images_orig_unblocked_v1_4_1
#python3.11 -u 7_bg_scores.py --sname generated_images_mitigated --simple_name

for i in $(seq 0 11);
do
    python3.11 -u 7_bg_scores.py --sname generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_0415
done

#pyiqa qalign -t generated_images_orig_blocked_r_cluster_"$i"_embeddings_block_all_0428 --device cuda -v > qalign_"$i".out
#9854-unblocked
#9850-blocked
#9851-mitigated
#9855-cluster:0-5
#9856-cluster:6-11