from pathlib import Path
from typing import Union, Optional
from loguru import logger
import hashlib


def download_file(weights_path: Union[Path, str], url: str, checksum: Optional[str] = None):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        logger.info(f"Downloading {url} to {weights_path}")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        import gdown

        gdown.download(
            url,
            str(weights_path),
            quiet=False,
        )
    else:
        logger.info(f"Skipping download of {url} to {weights_path} as file already exists")

    sha256 = hashlib.sha256()
    with weights_path.open("rb") as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)

    assert sha256.hexdigest() == checksum
