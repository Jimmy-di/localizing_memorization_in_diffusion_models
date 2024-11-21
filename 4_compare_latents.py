from utils.stable_diffusion import load_sd_components, load_text_components
from rtpt import RTPT
from tqdm.auto import tqdm
from utils.activation_detection import compute_noise_diff
import matplotlib.pyplot as plt
from torchmetrics.functional import multiscale_structural_similarity_index_measure
import torch
import pandas as pd
import argparse
import re
import numpy as np

def str_to_list(s):
    pattern = re.compile(r'\[.*?\]')
    sublists = pattern.findall(s)
    return [list(map(int, re.findall(r'\d+', sublist))) for sublist in sublists]


def main():
    args = create_parser()

    # Load SD components
    vae, unet, scheduler = load_sd_components(args.version)
    tokenizer, text_encoder = load_text_components(args.version)

    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    unmemorized_prompts = pd.read_csv(args.prompts, sep=';')['Caption'].tolist()
    refined_neurons_list = pd.read_csv(args.prompts, sep=';')['Refined Neurons'].tolist()

    rtpt = RTPT(args.user, 'Calculate_Pairwise_SSIM', len(unmemorized_prompts))
    rtpt.start()

    pairwise_ssim_per_prompt = []
    for i, prompt in enumerate(tqdm(unmemorized_prompts)):
        blocked_indices = str_to_list(refined_neurons_list[i])
        noise_diff = compute_noise_diff([0, [prompt]], tokenizer, text_encoder, unet, scheduler, seed=args.seed, blocked_indices=blocked_indices, scaling_factor=args.scaling_factor, samples_per_prompt=10, guidance_scale=7, num_inference_steps=50)
        pairwise_combination_indices = torch.combinations(torch.arange(len(noise_diff)), r=2)

        pairwise_ssim_scores = []
        for idx in range(0, len(pairwise_combination_indices), args.batch_size):
            input_1 = noise_diff[pairwise_combination_indices[idx:idx + args.batch_size][:,0]]
            input_2 = noise_diff[pairwise_combination_indices[idx:idx + args.batch_size][:,1]]
            ssim = multiscale_structural_similarity_index_measure(input_1, input_2, reduction='none', kernel_size=11, betas=(0.33, 0.33, 0.33))
            pairwise_ssim_scores.append(ssim.detach().cpu())

        pairwise_ssim_per_prompt.append(torch.cat(pairwise_ssim_scores))

        rtpt.step()
    #print(torch.stack(pairwise_ssim_per_prompt))
    np.savetxt("pairwise_ssim_{}.txt".format(args.scaling_factor), torch.stack(pairwise_ssim_per_prompt).numpy())
    torch.save(torch.stack(pairwise_ssim_per_prompt), args.output.replace(".pt", "") + '_' + args.version.replace("-", "_") + "4_c"+".pt")


def create_parser():
    parser = argparse.ArgumentParser(description='Calculate Pairwise SSIM Scores')

    parser.add_argument(
        '--prompts', 
        default='prompts/additional_laion_prompts.csv', 
        type=str, 
        help='The file from which the prompts are loaded to calculate the statistics (default: \'prompts/additional_laion_prompts.csv\').'
    )
    parser.add_argument(
        '--output', 
        default='pairwise_ssim_per_prompt.pt', 
        type=str, 
        help='The file to which the activation statistics are written (default: \'pairwise_ssim_per_prompt.pt\').'
    )
    parser.add_argument(
        '--seed',
        default=1,
        type=int,
        help='The seed used for the SD inference (default: 1).'
    )
    parser.add_argument(
        '--batch_size',
        default=45,
        type=int,
        help='The batch size used for calculating the pairwise SSIM score (default: 45).'
    )
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
    parser.add_argument('--scaling_factor', default=0.0, type=float, help='Scaling factor for the blocking of neurons')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()