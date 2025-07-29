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


import torch
import time

import math
import logging

import torch.nn.functional as F
import requests

import itertools
import urllib

from skimage import data, img_as_float
from skimage.metrics import structural_similarity

from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
    estimate_sigma,
)

import cv2
import torchvision

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
import clip

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


def post_filter(embeddings, dedup = False):
    to_remove = set()
    if dedup:
        to_remove = set(connected_components_dedup(embeddings))
    
    return to_remove

def main(args):


    
    dname = args.sname+'_faiss_100'
    #os.makedirs(os.path.dirname(dname), exist_ok=True)

    img_folder = args.sname

    _, _, files = next(os.walk(img_folder))
    file_count = len(files)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)
    clip_model = "ViT-B/32"
    indice_folder = os.path.realpath("./../laion400m-index")
    columns_to_return = ["url", "image_path", "caption", "NSFW"]

    model, preprocess = clip.load(clip_model, device=device)

    image_path = indice_folder + "/image.index"
    #text_path = indice_folder + "/text.index"
    hdf5_path = indice_folder + "/metadata.hdf5"

    image_index = faiss.read_index(image_path)
    #text_index = faiss.read_index(text_path)

    print("loading metadata...")
    hdf5_metadata = Hdf5Metadata(hdf5_path)

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }

    file_count = 500

    img_names = []
    faiss_descriptions = []
    faiss_score = []
    seg_ssim = []
    dest_name = []

   # python3.11 faiss2.py --sname generated_images_orig_blocked_v1_4_1
    for index in range(16, file_count):
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
            
            #img_name = "{}_{}.jpg".format(str(index), str(second_i))
            
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
            num_result_ids = 100

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
            result_indices = results[:nb_results]
            result_distances = distances[0][:nb_results]
            embeddings = embeddings[0][:nb_results]
            embeddings = normalized(embeddings)

            local_indices_to_remove = post_filter(embeddings, dedup=True)

            indices_to_remove = set()

            for local_index in local_indices_to_remove:
                indices_to_remove.add(result_indices[local_index])
        
            indices = []
            distances = []
            
            for ind, distance in zip(result_indices, result_distances):
                if ind not in indices_to_remove:
                    indices_to_remove.add(ind)
                    indices.append(ind)
                    distances.append(distance)

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

                    #im = im.resize((base_width, base_width), Image.Resampling.LANCZOS)
                    #print("reached here {}".format(i))
                    

                    img_names.append(img_name)
                    faiss_descriptions.append(result['caption'])
                    faiss_score.append(result['similarity'])
                    dest_name.append(save_img_name)


        print("{} done".format(index))
    
        dict = {'image name': img_names, 'caption': faiss_descriptions, 'score': faiss_score, 'destination': dest_name} 
    
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