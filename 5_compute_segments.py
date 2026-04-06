import argparse
import os
from pathlib import Path

from PIL import Image
from rembg import new_session, remove


def segment_image(session, im):
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


def main(args):
    os.environ["OMP_NUM_THREADS"] = "2"

    mem_folder = Path(args.mem_folder)
    fg_folder = mem_folder / "foreground"
    bg_folder = mem_folder / "background"
    fg_folder.mkdir(exist_ok=True)
    bg_folder.mkdir(exist_ok=True)

    session = new_session("birefnet-general", providers=["CUDAExecutionProvider"])

    for i in range(args.start, args.end):
        img_path = mem_folder / f"{i}.jpg"
        if not img_path.exists():
            continue
        print(i)
        image = Image.open(img_path)
        foreground, background = segment_image(session, image)
        foreground.save(fg_folder / f"{i}.png")
        background.save(bg_folder / f"{i}.png")

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment memorized images into foreground and background."
    )
    parser.add_argument("--mem_folder", type=str, default="memorized_images",
                        help="Folder containing memorized images (default: memorized_images)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index of images to process (default: 0)")
    parser.add_argument("--end", type=int, default=500,
                        help="End index (exclusive) of images to process (default: 500)")
    args = parser.parse_args()
    main(args)
