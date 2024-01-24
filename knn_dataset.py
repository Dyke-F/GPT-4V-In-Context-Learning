import os
import random
from ast import literal_eval
from collections.abc import MutableSequence
from pathlib import Path
from pprint import pprint
from typing import Dict, Tuple
import tiktoken
import json

import cv2
import pandas as pd
from tabulate import tabulate

import random

import torch
import torch.nn.functional as F
import numpy as np


PathLike = str | Path


##### k-nn helper funcs ######

def load_img_embeddings(file_path):
    feat_array = np.load(file_path, allow_pickle=True)

    if feat_array.dtype.kind != 'V':
        vector_len = feat_array[0][1].shape[0]
        struct = np.dtype([('name', 'U100'), ('vector', 'f4', (vector_len,))])
        feat_array = np.array([(name, vec) for name, vec in feat_array], dtype=struct)

    return feat_array


def get_topk_similar_base(query_vector_name, dataset_vectors_path, k=10):
    dataset_vectors = load_img_embeddings(dataset_vectors_path)

    full_vec_name = str(query_vector_name)+".png"
    query_vector = dataset_vectors[dataset_vectors["name"] == full_vec_name]["vector"].squeeze()
    query_vector = torch.tensor(query_vector, dtype=torch.float32)

    # remove the query_vector from the dataset_vectors
    filtered_dataset = dataset_vectors[dataset_vectors["name"] != full_vec_name]
    filtered_vectors = torch.tensor([vec for vec in filtered_dataset["vector"]], dtype=torch.float32) # slow

    query_vector /= query_vector.norm()
    filtered_vectors /= filtered_vectors.norm(dim=1, keepdim=True)

    cos_similarities = F.cosine_similarity(filtered_vectors, query_vector.unsqueeze(0), dim=1)
    topk_values, topk_indices = torch.topk(cos_similarities, k)
    topk_names_vectors = [(filtered_dataset[idx]["name"], filtered_dataset[idx]["vector"]) for idx in topk_indices]
    
    return topk_values, topk_names_vectors, filtered_dataset[topk_indices] # ignore the last output


def filter_by_label(dataset_vectors, label):
    # for multi-label k-nn few-shot sampling we need to filter the dataset_vectors by label
    # use Path(vec["name"]).name.startswith(label) because the "label" is the full path to the image
    return np.array([vec for vec in dataset_vectors if Path(vec["name"]).name.startswith(label)]) # TODO: implement for other datasets


def get_topk_similar_per_label(query_vector_name, dataset_vectors, use_only=None, k=10, most_similar_last=True, take_random=False):
    # full_vec_name = str(query_vector_name)+".png"
    query_vector = dataset_vectors[dataset_vectors["name"] == query_vector_name]["vector"].squeeze()
    query_vector = torch.tensor(query_vector, dtype=torch.float32)

    topk_sims_per_label = {}
    for label in use_only:
        subset_vectors = filter_by_label(dataset_vectors, label)

        # remove the query_vector from the dataset_vectors
        filtered_subsset = subset_vectors[subset_vectors["name"] != query_vector_name] # This line prevents data leakage DONT remove
        filtered_subvectors = torch.tensor(filtered_subsset["vector"], dtype=torch.float32)

        query_vector /= query_vector.norm()
        filtered_subvectors /= filtered_subvectors.norm(dim=1, keepdim=True)

        if take_random:
            filtered_subvectors = filtered_subvectors[torch.randperm(filtered_subvectors.size()[0])]
            random_imgs = [filtered_subsset[idx]["name"] for idx in range(k)]
            topk_sims_per_label[label] = random_imgs
        
        else:
            cos_similarities = F.cosine_similarity(filtered_subvectors, query_vector.unsqueeze(0), dim=1)
            topk_values, topk_indices = torch.topk(cos_similarities, k)
            # topk_names_vectors = [(filtered_subsset[idx]["name"], filtered_subsset[idx]["vector"]) for idx in topk_indices]
            topk_img_names = [filtered_subsset[idx]["name"] for idx in topk_indices]

            # topk_sims_per_label[label] = topk_values, topk_names_vectors, filtered_subsset[topk_indices]

            print("Most similar images and topk values are: ")
            if most_similar_last:
                topk_values = list(reversed(topk_values))
                topk_img_names = list(reversed(topk_img_names))
                topk_sims_per_label[label] = topk_img_names #, topk_values
                print(list(zip(topk_img_names, topk_values)))
            else:
                topk_sims_per_label[label] = topk_img_names #, topk_values
                print(list(zip(topk_img_names, topk_values)))
            print("----------")

    print(topk_sims_per_label)
    return topk_sims_per_label


class GPT4VKNNDataset:
    def __init__(
        self,
        datafile_path: PathLike,
        use_only: MutableSequence,
        dataset_vectors_path: PathLike,
        label_replacements: Dict = None,
        use_tiles=None,
        most_similar_last=True,
        take_random=False,
    ):
        data = pd.read_csv(datafile_path, converters={"path": literal_eval}, usecols=["fname", "label", "path"])
        data = data[data["label"].isin(use_only)]

        if use_tiles:
            data = data[data["fname"].str.contains(use_tiles)]

        self.data = data.sample(frac=1).reset_index(drop=True)  # shuffle
        self.label_replacements = label_replacements
        self.use_only = use_only
        self.dataset_vectors_path = dataset_vectors_path
        print("Loading dataset vectors...")
        self.dataset_vectors = load_img_embeddings(dataset_vectors_path)
        self.most_similar_last = most_similar_last
        self.take_random = take_random

    def __call__(self, num_shots: int = 0, show_bbox: bool = False):
        for i in range(len(self.data)):
            target_image_path = self.data.iloc[i]["path"][0] # this is where we will have to enter to make changes for multi image support !
            target_image_fname = self.data.iloc[i]["fname"]
            target_image_label = self.data.iloc[i]["label"]

            if num_shots == 0:
                yield {
                    "image_path": target_image_path,
                    "fname": target_image_fname,
                    "label": target_image_label,
                }
        
            if num_shots > 0:
                # in this implementation the test image is depleted from the query set

                topk_few_shot_samples = get_topk_similar_per_label(
                    query_vector_name=target_image_path,
                    dataset_vectors=self.dataset_vectors,
                    use_only=self.use_only,
                    k=num_shots,
                    most_similar_last=self.most_similar_last,
                    take_random=self.take_random,
                )

                multi_shot_mappings = {}
                for label, topk_samples in topk_few_shot_samples.items():
                    # remove cosine similarity values which are at ...[1]
                    # convert np.str_ to str
                    multi_shot_mappings[self.label_replacements[label]] = [str(tks) for tks in topk_samples]

                yield {
                    "image_path": target_image_path,
                    "fname": target_image_fname,
                    "label": target_image_label,
                    "multi_shot_mappings": {**multi_shot_mappings},
                }

                # TODO : implement for multiple input images


    def pprint_self(self):
        terminal_width = os.get_terminal_size().columns
        pd.set_option("display.width", terminal_width)
        print(tabulate(self.data, headers="keys", tablefmt="psql", showindex=False))
        # pd.reset_option('all')
