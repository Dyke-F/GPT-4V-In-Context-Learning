# utility functions with no specific use

from pathlib import Path
from typing import Dict, Tuple

import cv2
import pandas as pd

PathLike = str | Path


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
