import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean
from PIL import Image
import torch

from utils.stable_diffusion import generate_images
from utils.stable_diffusion import load_sd_components, load_text_components
from diffusers import DDIMScheduler, UNet2DConditionModel
from rtpt import RTPT

import pandas as pd


def aug_prompt(
    self,
    prompt=None,
    height=None,
    width=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt=None,
    num_images_per_prompt=1,
    eta=0.0,
    generator=None,
    latents=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    target_steps=[0],
    lr=0.1,
    optim_iters=10,
    target_loss=None,
    print_optim=False,
    optim_epsilon=None,
    alpha=0.5,
):

    return 0


def main():
    # load diffusion model
    args = create_parser()

    vae, unet, scheduler = load_sd_components(args.version, args.unet_id)
    tokenizer, text_encoder = load_text_components(args.version)

    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    df = pd.read_csv(args.result_file, sep=";")

    rtpt = RTPT(args.user, "image generation", len(df) // args.batch_size)
    rtpt.start()

    for i in tqdm(range(len(df) // args.batch_size), total=len(df) // args.batch_size):
        rows = df.iloc[i * args.batch_size : (i + 1) * args.batch_size]
        prompts = rows["Caption"].to_list()

        blocked_indices = None

        images = generate_images(
            [i, prompts],
            tokenizer,
            text_encoder,
            vae,
            unet,
            scheduler,
            num_inference_steps=args.num_steps,
            blocked_indices=blocked_indices,
            scaling_factor=args.scaling_factor,
            guidance_scale=args.guidance_scale,
            samples_per_prompt=args.num_samples,
            seed=args.seed
        )

        for j in range(len(images)):
            images[j].save(
                f"{args.output_path}/img_{i*args.batch_size + j // args.num_samples:04d}_{j%args.num_samples:02d}.jpg"
            )
        rtpt.step()


def create_parser():
    parser = argparse.ArgumentParser(description="Generating images")
    parser.add_argument(
        "-f",
        "--result_file",
        default="results/memorization_statistics_v1_4.csv",
        type=str,
        dest="result_file",
        help="path to file with image descriptions (default: results/memorization_statistics_v1_4.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="generated_images",
        type=str,
        dest="output_path",
        help="output folder for generated images (default: 'generated_images')",
    )
    parser.add_argument(
        "--unet_id",
        default=None,
        type=str,
        dest="unet_id",
        help="Using Local-stored Unet",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=2,
        type=int,
        dest="seed",
        help="seed for generated images (default: 2",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        default=10,
        type=int,
        dest="num_samples",
        help="number of generated samples for each prompt (default: 10)",
    )
    parser.add_argument(
        "--steps",
        default=50,
        type=int,
        dest="num_steps",
        help="number of denoising steps (default: 50)",
    )
    parser.add_argument(
        "-g",
        "--guidance_scale",
        default=7,
        type=float,
        dest="guidance_scale",
        help="guidance scale (default: 7)",
    )
    parser.add_argument(
        "-u",
        "--user",
        default="XX",
        type=str,
        dest="user",
        help='name initials for RTPT (default: "XX")',
    )
    parser.add_argument(
        "-v",
        "--version",
        default="v1-4",
        type=str,
        dest="version",
        help='Stable Diffusion version (default: "v1-4")',
    )
    parser.add_argument(
        "--original_images",
        action="store_true",
        default=False,
        help="Generate the original images",
    )

    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="Number of prompts per batch"
    )
    parser.add_argument(
        "--scaling_factor",
        default=0,
        type=float,
        help="Scaling factor for the blocking of neurons",
    )
    parser.add_argument("--optim_target_steps", default=0, type=int)
    parser.add_argument("--optim_lr", default=0.05, type=float)
    parser.add_argument("--optim_iters", default=10, type=int)
    parser.add_argument("--optim_target_loss", default=None, type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
