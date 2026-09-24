"""Download and unpack the dida test-task archive."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://cdn.dida.do/downloads/dida-test-task/dida_test_task.zip"


def download(dest: Path, url: str = DATA_URL) -> None:
    dest = Path(dest)
    image_dir = dest / "images"
    label_dir = dest / "labels"
    if image_dir.is_dir() and label_dir.is_dir() and any(image_dir.glob("*.png")):
        print(f"data already present in {dest}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "dida_test_task.zip"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if not name.startswith("__MACOSX")]
        zf.extractall(dest, members)
    archive.unlink()
    print(f"unpacked images and labels into {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data"))
    args = parser.parse_args()
    download(args.dest)


if __name__ == "__main__":
    main()
