import os
import random
from ast import literal_eval
from collections.abc import MutableSequence
from pathlib import Path
from pprint import pprint
from typing import Dict, Tuple
import tiktoken

import cv2
import pandas as pd
from tabulate import tabulate

PathLike = str | Path


# ============================== utility functions ============================== #

def get_files(directory: PathLike = "./data"):
    return Path(directory).rglob("*.png")


def image_info(
    image_path: PathLike, show: bool = False
) -> Dict[str, Tuple[PathLike, str, tuple]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"No image found at path {image_path}")

    if show:
        cv2.imshow("Image", image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "fname": Path(image_path).stem,
        "path": str(image_path),
        "shape": image.shape,
    }


def make_dataset_file(directory: str, file_name: str) -> None:
    files = get_files(directory)
    data = pd.DataFrame([image_info(f) for f in files])
    data["label"] = data["path"].apply(lambda x: str(x).rsplit("/")[-2])
    data.sort_values("fname", inplace=True)
    data = (
        data.groupby("fname")
        .agg({"label": "first", "shape": "first", "path": list})
        .reset_index()
    )
    data["path"] = data["path"].apply(lambda x: sorted(x))
    data.to_csv(f"{file_name}.csv", index=False)

# ============================== utility functions ============================== #


class GPT4VEvalDataset:
    def __init__(
        self,
        datafile_path: PathLike,
        use_only: MutableSequence,
        label_replacements: Dict = None,
        use_tiles=None,
    ):
        data = pd.read_csv(datafile_path, converters={"path": literal_eval}, usecols=["fname", "label", "path"])
        data = data[data["label"].isin(use_only)]

        if use_tiles:
            data = data[data["fname"].str.contains(use_tiles)]

        self.data = data.sample(frac=1).reset_index(drop=True)  # shuffle
        self.label_replacements = label_replacements

    def __call__(self, num_shots: int = 0, show_bbox: bool = False):
        for i in range(len(self.data)):
            target_image_path = self.data.iloc[i]["path"][0]
            target_image_fname = self.data.iloc[i]["fname"] # feed this into the get_topk_similar
            target_image_label = self.data.iloc[i]["label"]

            if num_shots == 0:
                yield {
                    "image_path": target_image_path,
                    "fname": target_image_fname,
                    "label": target_image_label,
                }

            if num_shots > 0:
                shot_df = self.data.copy(deep=True)
                shot_df.drop(i, inplace=True)  # prevent accidential sample leakage

                multi_shot_mappings = {}
                for label in shot_df["label"].unique():
                    multi_shot_mappings[self.label_replacements[label]] = (
                        shot_df[shot_df["label"] == label]
                        .sample(num_shots)["path"]
                        .to_list()
                    )

                if not show_bbox:
                    multi_shot_mappings = {
                        key: [sample[0] for sample in sl]
                        for key, sl in multi_shot_mappings.items()
                    }

                    yield {
                        "image_path": target_image_path,
                        "fname": target_image_fname,
                        "label": target_image_label,
                        "multi_shot_mappings": {**multi_shot_mappings},
                    }

                else:
                    multi_shot_mappings = {
                        key: [s for sl in ls for s in sl]
                        for key, ls in multi_shot_mappings.items()
                    }

                    yield {
                        "image_path": target_image_path,
                        "fname": target_image_fname,
                        "label": target_image_label,
                        "multi_shot_mappings": {**multi_shot_mappings},
                    }

    def pprint_self(self):
        terminal_width = os.get_terminal_size().columns
        pd.set_option("display.width", terminal_width)
        print(tabulate(self.data, headers="keys", tablefmt="psql", showindex=False))
        # pd.reset_option('all')


# calculate num tokens for text embeddings
def num_tokens_from_messages(messages, model="gpt-4-vision-preview", img_quality="high"):
    """Return the number of tokens used by a list of messages."""
    # THIS FUNCTION REQUIRES FIXES, but currently can be an approximate guess for # tokens used in 1 full message
    try:
        encoding = tiktoken.encoding_for_model(model)

    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in {
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1

    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613. Precise token counts for gpt-4-vision-preview and gpt-4-1106-preview need to be manually checked."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main for information on how messages are converted to tokens."""
        )

    def process_element(element, img_quality):
        encoded_tokens = 0
        if isinstance(element, dict):
            for key, value in element.items():
                if key == "image_url":
                    if img_quality == "low":
                        # TODO: FIXME
                        # THESE NUMBERS ARE NOT CORRECT AND JUST REPRESENT THE MAXIMUM NUMBER OF TOKENS FOR A SINGLE IMAGE
                        # WE WOULD RUN INTO IN THESE EXPERIMENTS
                        encoded_tokens += (
                            85 + 85
                        )  # https://platform.openai.com/docs/guides/vision
                    else:
                        encoded_tokens += (
                            170 * 6 + 85
                        )  # https://platform.openai.com/docs/guides/vision
                else:
                    encoded_tokens += len(key)
                    encoded_tokens += process_element(value, img_quality)
        elif isinstance(element, list):
            for item in element:
                encoded_tokens += process_element(item, img_quality)
        elif isinstance(element, str):
            encoded_tokens += len(encoding.encode(element))
        return encoded_tokens

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        num_tokens += process_element(message, img_quality)
    num_tokens += 3  # answer: <|start|>assistant<|message|>
    print("-------")
    return num_tokens
