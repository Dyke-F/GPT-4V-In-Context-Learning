import json
import os
from base64 import b64encode
from pathlib import Path
from pprint import pprint
from time import time
from typing import Dict, List, Union

import fsspec
from fastapi.encoders import jsonable_encoder
from omegaconf import OmegaConf
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential

from dataset import GPT4VEvalDataset
from knn_dataset import GPT4VKNNDataset
# from multi_image_knn_dataset import GPT4MultiImageKNNDataset

PathLike = str | Path


def _gen_system_message(system_prompt: str) -> Dict:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt,
            },
        ],
    }


def _gen_user_message(user_query: str, images: List[str], img_quality: str):
    assert isinstance(
        images, list
    ), "images must be a list, for single image use [image]"

    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_query,
            },
            *[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",
                        "detail": f"{img_quality}",
                    },
                }
                for image in images
            ],
        ],
    }


class GPT4V:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        user_query: str,
        mode: str,
        dataset: Union[GPT4VEvalDataset, GPT4VKNNDataset],
        img_quality: str = "high",
        model_kwargs: Dict = None,
        show_bbox: bool = False,
    ):
        self.client = OpenAI()
        self.model_name = model_name or "gpt-4-vision-preview"
        self.system_prompt = system_prompt
        self.user_query = user_query
        self.mode = mode
        self.encoded_images = self.batch_encode_images(dataset)
        #print(len(self.encoded_images))
        self.img_quality = img_quality
        self.model_kwargs = model_kwargs
        self.show_bbox = show_bbox
        self.cached_conversations = []
        self.interleaved_shots = True  # TODO: SET TO TRUE
        self.dataset = dataset

    def encode_image(self, image_path: PathLike) -> str:
        with fsspec.open(image_path, mode="rb") as img_file:
            return b64encode(img_file.read()).decode("utf-8")

    def batch_encode_images(self, dataset: GPT4VEvalDataset) -> None: # TODO improve type hints
        dataframe = dataset.data.copy(deep=True)
        image_paths = dataframe["path"].to_list()
        image_paths = set(
            [p for sublist in image_paths for p in sublist]
        )  # in every run we have the full batch of images and in the paths the bbox images as well
        encoded_images = {}
        for image_path in image_paths:
            encoded_images[image_path] = self.encode_image(image_path)
        return encoded_images

    def zero_shot(
        self,
        system_prompt: str,
        user_query: str,
        query_images: List,
        img_quality: str,
        **kwargs,  # ignore multi_shot additional keyword arguments
    ):
        messages = [_gen_system_message(system_prompt)]
        messages.append(_gen_user_message(user_query, query_images, img_quality))
        return messages

    def multi_shot(
        self,
        system_prompt: str,
        user_query: str,
        query_images: List,
        img_quality: str,
        multi_shot_mappings,
    ):
        messages = [_gen_system_message(system_prompt)]

        # TODO: find a nicer way to split the user query
        user_query_pre = user_query.split("-----------")[0].strip()
        user_query_post = user_query.split("-----------")[1].strip()

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_query_pre,
                    }
                ],
            }
        )

        if self.show_bbox:
            # for deprecated breast MRI image tests
            prompt = "Additionally, to help you find the tumor, we have provided the same image with a red bounding box around the tumor as the next image each: "
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            )

        if self.interleaved_shots:
            max_len = len(list(multi_shot_mappings.values())[0])
            for i in range(max_len):
                for j, instruct in enumerate(multi_shot_mappings.keys()):
                    if i < len(multi_shot_mappings[instruct]):
                        image = multi_shot_mappings[instruct][i]
                        image_content = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": f"{img_quality}",
                            },
                        }
                        examples = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruct},
                                image_content,
                            ],
                        }
                        messages.append(examples)

        else:
            for instruct, shot_images in multi_shot_mappings.items():
                image_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                            "detail": f"{img_quality}",
                        },
                    }
                    for image in shot_images
                ]
                examples = {
                    "role": "user",
                    "content": [{"type": "text", "text": instruct}, *image_content],
                }

                messages.append(examples)

        messages.append(_gen_user_message(user_query_post, query_images, img_quality))
        return messages

    def preprocess_inputs(self, sample: Dict) -> List[Dict]:
        init_messages = getattr(self, self.mode)
        assert init_messages is not None, f"Incorrect mode: {self.mode}"
        # single image mode
        if isinstance(sample["image_path"], str): # couldve also checked type (str vs list or so)
            query_img = self.encoded_images[sample["image_path"]]
        # else multi image mode
        else:
            query_img = [self.encoded_images[img_path] for img_path in sample["image_path"]]
        multi_shot_mappings = sample.get("multi_shot_mappings", None)

        # here we only b64-encoded images that are test images
        # for KNN-sampling we encode at runtime, as encoding 100k+ images leads to overhead while ...
        # not requiring the majority of the images anyways
        if multi_shot_mappings is not None:
            if self.dataset.__class__.__name__ == "GPT4VEvalDataset":
                multi_shot_mappings = {
                    key: [self.encoded_images[p] for p in paths]
                    for key, paths in multi_shot_mappings.items()
                }
            else: # else its an GPT4VKNNDataset or a GPT4MultiImageKNNDataset -> encode at runtime
                multi_shot_mappings = {
                    key: [self.encode_image(str(p)) for p in paths]
                    for key, paths in multi_shot_mappings.items()
                }

        messages = init_messages(
            system_prompt=self.system_prompt,
            user_query=self.user_query,
            query_images=[query_img] if isinstance(query_img, str) else query_img,
            img_quality=self.img_quality,
            multi_shot_mappings=multi_shot_mappings,
        )

        return messages

    @retry(wait=wait_fixed(15), stop=stop_after_attempt(10))
    def run_model(self, messages: List[Dict]):
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=4096,
            seed=42,
            temperature=0.1, # TODO: check
            # response_format={ "type": "json_object" },
            # **self.model_kwargs,
        )
        return response

    def __repr__(self):
        return f"GPT4V(model_name={self.model_name}, mode={self.mode}, img_quality={self.img_quality})"

    def prepare_for_save(
        self, response: Dict, sample: Dict, data_cfg: Dict, user_args: Dict
    ):
        to_save = {}
        to_save["unixtime"] = int(time())
        to_save["system_prompt"] = self.system_prompt
        to_save["user_query"] = self.user_query
        to_save["model_name"] = self.model_name
        to_save["mode"] = self.mode
        to_save["img_quality"] = self.img_quality
        to_save["model_kwargs"] = jsonable_encoder(self.model_kwargs)
        to_save["full_response"] = jsonable_encoder(response)
        to_save["model_response"] = json.dumps(response.choices[0].message.content)

        # TODO: CHECK THIS IMPLEMENTATION
        if not isinstance(sample["label"], str):
            sample["label"] = int(sample["label"])

        to_save["sample"] = sample
        to_save["label"] = sample["label"]
        to_save["fname"] = sample["fname"]
        to_save["data_cfg"] = json.dumps(OmegaConf.to_container(data_cfg, resolve=True))
        to_save["user_args"] = json.dumps(
            OmegaConf.to_container(user_args, resolve=True)
        )

        self.cached_conversations.append(to_save)

    def save_conversation(self, save_path: PathLike):
        assert not os.path.exists(save_path), f"File already exists: {save_path}"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with fsspec.open(f"{save_path}.json", "w") as file:
            json.dump(self.cached_conversations, file)

    def __call__(
        self,
        sample: Dict,
        verbose: bool = False,
    ):
        messages = self.preprocess_inputs(sample)
        response = self.run_model(messages)
        if verbose:
            pprint(response.choices[0].message.content)
        self.prepare_for_save(response)
