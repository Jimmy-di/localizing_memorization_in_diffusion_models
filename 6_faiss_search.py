"""Clip back: host a knn service using clip as an encoder"""
import inspect

from typing import Callable, Dict, Any, List
import faiss
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import json
from io import BytesIO
from PIL import Image
import base64
import ssl
import os
import fire
from pathlib import Path
import pandas as pd
import urllib
import tempfile
import io
import numpy as np
from functools import lru_cache
import pyarrow as pa
import fsspec
import argparse
from functools import partial

import time
import math
import logging
import torch
import torch.nn.functional as F
import requests
from mmseg.apis import init_segmentor, inference_segmentor
import itertools
import urllib

import mmcv
from mmcv.runner import load_checkpoint
from skimage import data, img_as_float
from skimage.metrics import structural_similarity

from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
    estimate_sigma,
)

import cv2

from clip_retrieval.ivf_metadata_ordering import (
    Hdf5Sink,
    external_sort_parquet,
    get_old_to_new_mapping,
    re_order_parquet,
)
from dataclasses import dataclass
from all_clip import load_clip  # pylint: disable=import-outside-toplevel

from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import h5py

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/dinov2'
sys.path.insert(0, parentdir) 

import dinov2.eval.segmentation.models
import dinov2.eval.segmentation.utils.colormaps as colormaps


class Hdf5Metadata:

    def __init__(self, hdf5_file):
        f = h5py.File(hdf5_file, "r")
        self.ds = f["dataset"]

    def get(self, ids, cols=None):
        """implement the get method from the hdf5 metadata provide, get metadata from ids"""
        items = [{} for _ in range(len(ids))]
        if cols is None:
            cols = self.ds.keys()
        else:
            cols = list(self.ds.keys() & set(cols))
        for k in cols:
            for i, e in enumerate(ids):
                items[i][k] = self.ds[k][e]
        return items

def convert_metadata_to_base64(meta):
    """
    Converts the image at a path to the Base64 representation and sets the Base64 string to the `image`
    key in the metadata dictionary.
    If there is no `image_path` key present in the metadata dictionary, the function will have no effect.
    """
    if meta is not None and "image_path" in meta:
        path = meta["image_path"]
        if os.path.exists(path):
            img = Image.open(path)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            meta["image"] = img_str

def meta_to_dict(meta):
    output = {}
    for k, v in meta.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif type(v).__module__ == np.__name__:
            v = v.item()
        output[k] = v
    return output

def connected_components(neighbors):
    """find connected components in the graph"""
    seen = set()

    def component(node):
        r = []
        nodes = set([node])
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= set(neighbors[node]) - seen
            r.append(node)
        return r

    u = []
    for node in neighbors:
        if node not in seen:
            u.append(component(node))
    return u

def get_non_uniques(embeddings, threshold=0.94):
    """find non-unique embeddings"""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)  # pylint: disable=no-value-for-parameter
    l, _, I = index.range_search(embeddings, threshold)  # pylint: disable=no-value-for-parameter,invalid-name

    same_mapping = defaultdict(list)

        # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
    for i in range(embeddings.shape[0]):
        for j in I[l[i] : l[i + 1]]:
            same_mapping[int(i)].append(int(j))

    groups = connected_components(same_mapping)
    non_uniques = set()
    for g in groups:
        for e in g[1:]:
            non_uniques.add(e)

    return list(non_uniques)

def connected_components_dedup(embeddings):
    non_uniques = get_non_uniques(embeddings)
    return non_uniques




def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")

DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}



def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_segmenter(cfg, backbone_model):
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model

def render_segmentation(segmentation_logits, dataset):
        colormap = DATASET_COLORMAPS[dataset]
        colormap_array = np.array(colormap, dtype=np.uint8)
        segmentation_values = colormap_array[segmentation_logits + 1]
        return Image.fromarray(segmentation_values)


def main(args):


    
    dname = args.sname+'_faiss_20'
    #os.makedirs(os.path.dirname(dname), exist_ok=True)

    img_folder = args.sname
    seg_folder = args.sname + '_segmentations'

    _, _, files = next(os.walk(img_folder))
    file_count = len(files)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = "ViT-B/32"
    indice_folder = os.path.realpath("./../laion400m-index")
    columns_to_return = ["url", "image_path", "caption", "NSFW"]

    model, preprocess, tokenizer = load_clip(clip_model, device=device)

    image_path = indice_folder + "/image.index"
    #text_path = indice_folder + "/text.index"
    hdf5_path = indice_folder + "/metadata.hdf5"

    image_index = faiss.read_index(image_path)
    #text_index = faiss.read_index(text_path)

    print("loading metadata...")
    hdf5_metadata = Hdf5Metadata(hdf5_path)

    BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")


    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.cuda()

    DATASET_COLORMAPS = {
        "ade20k": colormaps.ADE20K_COLORMAP,
        "voc2012": colormaps.VOC2012_COLORMAP,
    }


    HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
    HEAD_DATASET = "ade20k" # in ("ade20k", "voc2012")
    HEAD_TYPE = "ms" # in ("ms, "linear")


    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    if HEAD_TYPE == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

    seg_model = create_segmenter(cfg, backbone_model=backbone_model)
    load_checkpoint(seg_model, head_checkpoint_url, map_location="cpu")
    seg_model.cuda()
    seg_model.eval()

    file_count = 500

    img_names = []
    faiss_descriptions = []
    faiss_score = []
    seg_ssim = []
    dest_name = []

   # 37 
   # 62
   # python3.11 faiss2.py --sname generated_images_orig_blocked_v1_4_1
    for index in range(0, file_count):
        for second_i in range(0,3):
        
            #img_name = "{}_{}.jpg".format(str(index), str(second_i))

            if index < 10:
                img_name = "img_000{}_0{}.jpg".format(str(index), str(second_i))
            elif index < 100:
                img_name = "img_00{}_0{}.jpg".format(str(index), str(second_i))
            elif index < 1000:
                img_name = "img_0{}_0{}.jpg".format(str(index), str(second_i))
            else:
                img_name = "img_{}_0{}.jpg".format(str(index), str(second_i))
            
            img_name = "{}_{}.jpg".format(str(index), str(second_i))
            
            try:
                img = Image.open(img_folder + '/' + img_name)
            except FileNotFoundError:
                continue            
            
            cv_img = np.array(img)
            cv_img = cv_img[:, :, ::-1]

            prepro = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(prepro)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query = image_features.cpu().to(torch.float32).detach().numpy()

            # Perform KNN
            num_result_ids = 50

            distances, indices, embeddings = image_index.search_and_reconstruct(query, num_result_ids)

            results = indices[0]
            nb_results = np.where(results == -1)[0]

            def normalized(a, axis=-1, order=2):
                l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
                l2[l2 == 0] = 1
                return a / np.expand_dims(l2, axis)


            if len(nb_results) > 0:
                nb_results = nb_results[0]
            else:
                nb_results = len(results)
            indices = results[:nb_results]
            distances = distances[0][:nb_results]
            embeddings = embeddings[0][:nb_results]
            embeddings = normalized(embeddings)

            if len(distances) == 0:
                print("NO MATCHING RESULTS!")
            

            num_images = len(indices)

            results = []

            metas = hdf5_metadata.get(indices[:num_images], columns_to_return)
            for key, (d, i) in enumerate(zip(distances, indices)):
                output = {}
                meta = None if key + 1 > len(metas) else metas[key]
                convert_metadata_to_base64(meta)
                if meta is not None:
                    output.update(meta_to_dict(meta))
                output["id"] = i.item()
                output["similarity"] = d.item()
                results.append(output)

            #print(results)

            base_width = 512
            for i, result in enumerate(results):
                image_url = result['url']
                endings = image_url.split(".")[-1]
                if endings == "png":
                    endings = ".png"
                elif endings == 'jpeg':
                    endings = ".jpeg"
                else:
                    endings = ".jpg"
                try:
                    
                    im = Image.open(requests.get(image_url, stream=True, timeout=2).raw)
                    time.sleep(2)
                    #print("reached here2")
                except:
                    im = None
                    continue
                if im:
                    im = im.convert("RGB")
                    #print("reached here3")
                    im = im.resize((base_width, base_width), Image.Resampling.LANCZOS)
                    save_img_name = 'img_{}_{}_{}'.format(str(index), str(second_i), str(i))
                    im.save('./'+ dname + '/' + save_img_name + endings)
                    array = np.array(im)[:, :, ::-1].copy()

                    #im = img_as_float(im)
                    #im = denoise_tv_chambolle(im, weight=0.2, channel_axis=-1)
                    #im = denoise_bilateral(im, sigma_color=0.1, sigma_spatial=15, channel_axis=-1)
                    #im = Image.fromarray((im * 255).astype(np.uint8))
                    

                    segmentation_logits = inference_segmentor(seg_model, array)[0]
                    segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET).convert("L")
                    
                    #print("reached here {}".format(i))
            
                    segmented_image.save('./segments5/'+ save_img_name + endings)
                    smap = img_as_float(segmented_image)

                    orig_map = Image.open(seg_folder + '/' + img_name)
                    orig_map = img_as_float(orig_map)

                    ssim_noise = ssim(smap, orig_map, data_range=orig_map.max() - orig_map.min())

                    #im = im.resize((base_width, base_width), Image.Resampling.LANCZOS)
                    #print("reached here {}".format(i))
                    

                    img_names.append(img_name)
                    faiss_descriptions.append(result['caption'])
                    faiss_score.append(result['similarity'])
                    dest_name.append(save_img_name)
                    seg_ssim.append(ssim_noise)

        print("{} done".format(index))
    
        dict = {'image name': img_names, 'caption': faiss_descriptions, 'score': faiss_score, 'destination': dest_name, 'segment score': seg_ssim} 
    
        df = pd.DataFrame(dict)
        df.to_csv('./'+dname+'/report.csv', mode='a',  header=False, index=False)
        img_names = []
        faiss_descriptions = []
        faiss_score = []
        seg_ssim = []
        dest_name = []
    print(df) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--sname', type=str, required=True)
    # Parse the argument
    args = parser.parse_args()    
    main(args)