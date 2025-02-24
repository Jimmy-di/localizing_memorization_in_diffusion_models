"""Clip back: host a knn service using clip as an encoder"""


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

import h5py
from tqdm import tqdm
from prometheus_client import Histogram, REGISTRY, make_wsgi_app
import math
import logging
import torch 
import requests

from skimage.metrics import structural_similarity
import cv2

from clip_retrieval.ivf_metadata_ordering import (
    Hdf5Sink,
    external_sort_parquet,
    get_old_to_new_mapping,
    re_order_parquet,
)
from dataclasses import dataclass
from all_clip import load_clip  # pylint: disable=import-outside-toplevel

from image_similarity_measures.quality_metrics import ssim
from skimage.metrics import structural_similarity as ssim_2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

img_folder = "./generated_images_orig_blocked_r_cluster_2_embeddings_block_all_g3_0.428"
_, _, files = next(os.walk(img_folder))
file_count = len(files) // 3

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = "ViT-B/32"
indice_folder = os.path.realpath("./laion-400m")
columns_to_return = ["url", "image_path", "caption", "NSFW"]

model, preprocess, tokenizer = load_clip(clip_model, device=device)

image_path = indice_folder + "/image.index"
#text_path = indice_folder + "/text.index"
hdf5_path = indice_folder + "/metadata.hdf5"

image_index = faiss.read_index(image_path)
#text_index = faiss.read_index(text_path)

print("loading metadata...")
hdf5_metadata = Hdf5Metadata(hdf5_path)

file_count = 25
for index in range(file_count):
    for second_i in range(0,5):
    
        if index < 10:
            img = Image.open(img_folder + "/img_000{}_0{}.jpg".format(str(index), str(second_i))).convert('RGB')
        else:
            img = Image.open(img_folder + "/img_00{}_0{}.jpg".format(str(index), str(second_i))).convert('RGB')
        
        cv_img = np.array(img)
        cv_img = cv_img[:, :, ::-1]

        prepro = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(prepro)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        query = image_features.cpu().to(torch.float32).detach().numpy()

        # Perform KNN
        num_result_ids = 200

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

        """    localized_to_remove = connected_components_dedup(embeddings)

        r_indices = []
        r_distances = []

        indices_to_remove = set()
        for local_index in localized_to_remove:
            indices_to_remove.add(indices[local_index])

        for ind, distance in zip(indices, distances):
            if ind not in indices_to_remove:
                indices_to_remove.add(ind)
                r_indices.append(ind)
                r_distances.append(distance)

        indices = r_indices
        distances = r_distances
        """
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

        print(results)

        base_width = 512
        for result in results:
            image_url = result['url']
            endings = image_url.split(".")[-1]
            if endings == "png":
                endings = ".png"
            elif endings == 'jpeg':
                endings = ".jpeg"
            else:
                endings = ".jpg"
            try:
                im = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

                #img_data = requests.get(image_url)
                #time.sleep(2)
                #img_data = img_data.content
                #with open('./temp_image/image_name_{}_{}_{:.4f}_{}'.format(str(index), str(second_i), result['similarity'], result['id']) + endings, 'wb') as handler:
                #    handler.write(img_data)
            except:
                im = None
                continue
            
            w, h = im.size
            im = im.resize((base_width, base_width), Image.Resampling.LANCZOS)
            cv_im = np.array(im)
            cv_im = cv_im[:, :, ::-1]

            # Convert images to grayscale
            image1_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
            # Calculate SSIM
            ssim_score = structural_similarity(image1_gray, image2_gray, full=True)
            # print(f"SSIM Score: ", round(ssim_score[0], 2))
            if ssim_score[0] > 0.4:
                im.save('./temp_image_2/{}_{}_{}_{}'.format(str(index), str(second_i), str(round(ssim_score[0], 3)), str(round(result['similarity'], 3))) + endings)
        #del cv_im
    #del cv_img   

            #if im:
            #    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            #    opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)