import argparse
import os
from pathlib import Path
from queue import PriorityQueue

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rembg import new_session, remove
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchvision import transforms


def fb_strip(session, im):
    foreground = remove(
        im,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=270,
        alpha_matting_background_threshold=20,
        alpha_matting_erode_size=11,
    )
    alpha_chan = foreground.split()[-1].convert("L")
    background = im.copy()
    background.paste(alpha_chan, mask=alpha_chan)
    return foreground, background


def to_bchw(image):
    tensor = transforms.ToTensor()(image)
    while len(tensor.shape) < 4:
        tensor = tensor.unsqueeze(0)
    return tensor


def is_mostly_white(image, threshold=0.97, white=True):
    pixels = np.array(image) / 255.0
    count = np.sum(pixels > 0.97) if white else np.sum(pixels < 0.03)
    return count / pixels.size > threshold


def get_or_segment(session, gen_folder, fg_folder, bg_folder, img_name):
    fg_path = fg_folder / img_name
    bg_path = bg_folder / img_name
    if fg_path.exists() and bg_path.exists():
        return Image.open(fg_path).convert("RGBA"), Image.open(bg_path).convert("RGBA")
    gen_img = Image.open(gen_folder / img_name)
    fg, bg = fb_strip(session, gen_img)
    fg.convert("RGB").save(fg_path)
    bg.convert("RGB").save(bg_path)
    return fg.convert("RGBA"), bg.convert("RGBA")


def compute(args):
    mem_folder = Path(args.mem_folder)
    gen_folder = Path(args.gen_folder)
    fg_folder = Path(f"{args.gen_folder}_foreground")
    bg_folder = Path(f"{args.gen_folder}_background")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ms_ssim = (StructuralSimilarityIndexMeasure() if args.ssim
               else MultiScaleStructuralSimilarityIndexMeasure()).to(device)

    session = None if args.no_segment else new_session(
        "birefnet-general", providers=["CUDAExecutionProvider"]
    )
    if not args.no_segment:
        fg_folder.mkdir(exist_ok=True)
        bg_folder.mkdir(exist_ok=True)

    out_csv = gen_folder / ("scores_no_seg.csv" if args.no_segment else "scores.csv")
    first_write = True
    all_scores = []

    for cluster in range(args.num_clusters):
        for k in range(args.num_prompts):
            orig_img, generated_img, bg_scores, fg_scores, avg_scores = [], [], [], [], []

            for j in range(args.num_samples):
                if args.no_cluster_prefix:
                    img_name = f"img_{k:04d}_{j:02d}.jpg"
                else:
                    img_name = f"{cluster}_img_{k:04d}_{j:02d}.jpg"
                q = PriorityQueue(maxsize=args.top_k)

                for i in range(args.num_mem_images):
                    imname = f"{i}.jpg" if args.no_segment else f"{i}.png"
                    im_path = mem_folder / imname
                    bg_path = mem_folder / "background" / imname
                    fg_path = mem_folder / "foreground" / imname

                    try:
                        if args.no_segment:
                            mem_t = to_bchw(Image.open(im_path).convert("RGBA").resize(
                                (512, 512), resample=Image.Resampling.LANCZOS))
                            mem_fg = mem_bg = mem_t
                            gen_t = to_bchw(Image.open(gen_folder / img_name).convert("RGBA"))
                            gen_fg = gen_bg = gen_t
                        else:
                            mem_bg = to_bchw(Image.open(bg_path).convert("RGBA").resize(
                                (512, 512), resample=Image.Resampling.LANCZOS))
                            mem_fg = to_bchw(Image.open(fg_path).convert("RGBA").resize(
                                (512, 512), resample=Image.Resampling.LANCZOS))
                            gen_fg_img, gen_bg_img = get_or_segment(
                                session, gen_folder, fg_folder, bg_folder, img_name)
                            gen_fg = to_bchw(gen_fg_img)
                            gen_bg = to_bchw(gen_bg_img)

                            if is_mostly_white(Image.open(bg_folder / img_name).convert("L")):
                                gen_bg = gen_fg
                                print("Using only foreground.")
                            elif is_mostly_white(Image.open(fg_folder / img_name).convert("L"), white=False):
                                gen_fg = gen_bg
                                print("Using only background.")
                    except Exception as e:
                        print(e)
                        continue

                    mem_bg, mem_fg = mem_bg.to(device), mem_fg.to(device)
                    gen_bg, gen_fg = gen_bg.to(device), gen_fg.to(device)

                    fscore = ms_ssim(mem_fg, gen_fg).cpu().item()
                    bscore = fscore if args.no_segment else ms_ssim(mem_bg, gen_bg).cpu().item()
                    score = 0.5 * fscore + 0.5 * bscore

                    entry = (score, bscore, fscore, imname, img_name)
                    if not q.full():
                        q.put(entry)
                    else:
                        worst = q.get()
                        q.put(entry if score > worst[0] else worst)

                while not q.empty():
                    score, bscore, fscore, imname, img_name_ = q.get()
                    print(f"  score={score:.4f}  mem={imname}  gen={img_name_}")
                    bg_scores.append(bscore)
                    fg_scores.append(fscore)
                    avg_scores.append(score)
                    orig_img.append(imname)
                    generated_img.append(img_name_)

            df = pd.DataFrame({
                "original image": orig_img,
                "generated image": generated_img,
                "Background Scores": bg_scores,
                "Foreground Scores": fg_scores,
                "Average Scores": avg_scores,
            })
            df.to_csv(out_csv, index=False, mode="w" if first_write else "a", header=first_write)
            first_write = False
            all_scores.extend(avg_scores)
            print(f"cluster={cluster} k={k}: wrote {len(df)} rows")

    return all_scores


def main(args):
    os.environ["OMP_NUM_THREADS"] = "2"
    scores = compute(args)
    print(f"Done. Total scored pairs: {len(scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute segment-aware similarity scores between generated and memorized images."
    )
    parser.add_argument("--gen_folder", type=str, required=True,
                        help="Folder of generated images")
    parser.add_argument("--mem_folder", type=str, default="memorized_images",
                        help="Folder of memorized original images (default: memorized_images)")
    parser.add_argument("--num_clusters", type=int, default=12,
                        help="Number of clusters (default: 12)")
    parser.add_argument("--num_prompts", type=int, default=66,
                        help="Number of prompts per cluster (default: 66)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Generated samples per prompt (default: 5)")
    parser.add_argument("--num_mem_images", type=int, default=500,
                        help="Number of memorized images to compare against (default: 500)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Keep top-k most similar memorized images per generated image (default: 5)")
    parser.add_argument("--no_segment", action="store_true",
                        help="Skip foreground/background segmentation, compare full images")
    parser.add_argument("--ssim", action="store_true",
                        help="Use SSIM instead of MS-SSIM")
    parser.add_argument("--no_cluster_prefix", action="store_true",
                        help="Use img_XXXX_YY.jpg naming (step 4 default) instead of {cluster}_img_XXXX_YY.jpg")
    args = parser.parse_args()
    main(args)
