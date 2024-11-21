import os
import random

import torch
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from utils.stable_diffusion import generate_images
from utils.stable_diffusion import load_sd_components, load_text_components
import argparse
from utils.datasets import load_prompts
from rtpt import RTPT
import pandas as pd
import json
import re
import sys
from random import sample 
import numpy as np

def str_to_list(neuron_list, s):
    pattern = re.compile(r'\[.*?\]')
    sublists = pattern.findall(s)
    blocked_neurons_for_prompt = [list(map(int, re.findall(r'\d+', sublist))) for sublist in sublists]
    for i, sublist in enumerate(blocked_neurons_for_prompt):
        for neuron in sublist:
            if neuron not in neuron_list[i]:
                neuron_list[i].append(neuron)
    return neuron_list
"""    
def str_to_list(s):
    pattern = re.compile(r'\[.*?\]')
    sublists = pattern.findall(s)
    return [list(map(int, re.findall(r'\d+', sublist))) for sublist in sublists]
"""
def get_num_neurons_per_layer(unet):
    num_neurons = []
    for layer_idx in range(7):
        if layer_idx < 6:
            num_neurons.append(unet.down_blocks[int(layer_idx / 2)].attentions[layer_idx % 2].transformer_blocks[0].attn2.to_v.out_features )
        else:
            num_neurons.append(unet.mid_block.attentions[0].transformer_blocks[0].attn2.to_v.out_features )
    return num_neurons

@torch.no_grad()
def main():
    args = create_parser()

    vae, unet, scheduler = load_sd_components(args.version)
    tokenizer, text_encoder = load_text_components(args.version)
    
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    os.makedirs(args.output_path, exist_ok=False)
    
    with open(os.path.join(args.output_path, "config.json"), "w") as outfile:
        args_to_save = vars(args)
        args_to_save['command'] = " ".join(sys.argv)
        json.dump(args_to_save, outfile)
    
    # only one flag allowed
    assert not (args.initial_neurons and args.refined_neurons and args.original_images and args.block_random_neurons is not None)
    # only one flag allowed
    assert not (args.block_top_k_neurons_per_layer is not None and args.block_top_k_neurons is not None and args.block_top_k_neuron_subgroups is not None and args.block_random_neurons is not None)
    # assert that either the initial neurons or the refined neurons are chosen when blocking the top k neurons
    assert (
        ((args.block_top_k_neurons_per_layer is not None or args.block_top_k_neurons is not None or args.block_top_k_neuron_subgroups is not None) and args.initial_neurons) or 
        ((args.block_top_k_neurons_per_layer is not None or args.block_top_k_neurons is not None or args.block_top_k_neuron_subgroups is not None) and args.refined_neurons) or 
        (args.initial_neurons or args.refined_neurons) or
        args.original_images or
        args.block_random_neurons is not None
    ), "Either the initial neurons or the refined neurons must be chosen when blocking the top k neurons"
    
    # load csv file
    df = pd.read_csv(args.result_file, sep=';')
    
    # filter for vm or tm prompts
    blocked_indices_all = [[],[],[],[],[],[],[]]

    #for index, row in df.iterrows():
    #    blocked_indices_all = str_to_list(blocked_indices_all, row['Refined Neurons'])

    save_cluster = args.result_file[re.search(r"\d+\d*", args.result_file).start()]
    
    print(save_cluster)
    #np.savez("results2/neurons_cluster_{}.npz".format(save_cluster), blocked_indices_all[0],
    # blocked_indices_all[1],blocked_indices_all[2],blocked_indices_all[3],blocked_indices_all[4],
    # blocked_indices_all[5],blocked_indices_all[6])
    npz = np.load("results2/neurons_cluster_{}.npz".format(save_cluster))
    blocked_indices_all = [npz['arr_0'], npz['arr_1'], npz['arr_2'], 
                            npz['arr_3'], npz['arr_4'], npz['arr_5'], npz['arr_6']]
    #with open(FILEPATH, "rb") as jsonfile:
    #    blocked_indices_all = json.load(jsonfile)

    rtpt = RTPT(args.user, 'image generation', len(df) // args.batch_size)
    rtpt.start()
    for i in tqdm(range(len(df) // args.batch_size), total=len(df) // args.batch_size):
        rows = df.iloc[i*args.batch_size:(i+1)*args.batch_size]
        prompts = rows['Caption'].to_list()

        if args.block_top_k_neurons_per_layer is not None or args.block_top_k_neurons is not None or args.block_top_k_neuron_subgroups is not None or args.block_random_neurons or args.block_random_neuron_subgroups:   
            pass
        elif args.initial_neurons:
            blocked_indices = str_to_list(rows.iloc[0]['Initial Neurons'])
        elif args.refined_neurons:
            blocked_indices = blocked_indices_all
        elif args.original_images:
            blocked_indices = None
                    
        images = generate_images([i, prompts], tokenizer, text_encoder, vae, unet, scheduler, num_inference_steps=args.num_steps, blocked_indices=blocked_indices, scaling_factor=args.scaling_factor, guidance_scale=args.guidance_scale, samples_per_prompt=args.num_samples, seed=args.seed)

        for j in range(len(images)):
            images[j].save(f"{args.output_path}/img_{i*args.batch_size + j // args.num_samples:04d}_{j%args.num_samples:02d}.jpg")
        rtpt.step()

def create_parser():
    parser = argparse.ArgumentParser(description='Generating images')
    parser.add_argument(
        '-f',
        '--result_file',
        default='results/memorization_statistics_v1_4.csv',
        type=str,
        dest="result_file",
        help='path to file with image descriptions (default: results/memorization_statistics_v1_4.csv)')
    parser.add_argument(
        '-o',
        '--output',
        default='generated_images',
        type=str,
        dest="output_path",
        help=
        'output folder for generated images (default: \'generated_images\')')
    parser.add_argument('-s',
                        '--seed',
                        default=2,
                        type=int,
                        dest="seed",
                        help='seed for generated images (default: 2')
    parser.add_argument(
        '-n',
        '--num_samples',
        default=10,
        type=int,
        dest="num_samples",
        help='number of generated samples for each prompt (default: 10)')
    parser.add_argument('--steps',
                        default=50,
                        type=int,
                        dest="num_steps",
                        help='number of denoising steps (default: 50)')
    parser.add_argument('-g',
                        '--guidance_scale',
                        default=7,
                        type=float,
                        dest="guidance_scale",
                        help='guidance scale (default: 7)')
    parser.add_argument('-u',
                        '--user',
                        default='XX',
                        type=str,
                        dest="user",
                        help='name initials for RTPT (default: "XX")')
    parser.add_argument('-v',
                        '--version',
                        default='v1-4',
                        type=str,
                        dest="version",
                        help='Stable Diffusion version (default: "v1-4")')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='Number of prompts per batch')
    parser.add_argument('--original_images', action='store_true', default=False, help='Generate the original images')
    parser.add_argument('--initial_neurons', action='store_true', default=False, help='Block initial neurons')
    parser.add_argument('--refined_neurons', action='store_true', default=False, help='Block refined neurons')
    parser.add_argument('--block_top_k_neurons_per_layer', default=None, type=int, help='Blocks the top k found neurons for each layer for all the memorized sampels')
    parser.add_argument('--block_top_k_neurons', default=None, type=int, help='Blocks the top k found neurons over all layers for all the memorized sampels')
    parser.add_argument('--block_top_k_neuron_subgroups', default=None, type=int, help='Blocks the top k found neuron subgroups for all the memorized sampels')
    parser.add_argument('--block_random_neuron_subgroups', default=None, type=int, help='Blocks random neurons based on what the subgroups for the memorized samples were found.')
    parser.add_argument('--block_random_neurons', default=None, type=int, help='Blocks random neurons throughout all layers')
    parser.add_argument('--unmemorized_prompts', default=None, type=str, help='Path to the unmemorized prompt files. If set, the unmemorized prompts will be used instead of the memorized prompts. Only usable when blocking the top k or random neurons')
    parser.add_argument('--memorization_type', default=None, type=str, help='Decide if the neurons of the verbatim or template prompts should be used')
    parser.add_argument('--scaling_factor', default=0.0, type=float, help='Scaling factor for the blocking of neurons')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()