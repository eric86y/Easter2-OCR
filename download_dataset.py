"""
A very simple interface to download a dataset from BDRC's or my Huggingface site.
Check https://huggingface.co/BDRC or https://huggingface.co/Eric-23xd for alternative datasets.

run e.g.: python download_dataset.py --dataset "BDRC/Karmapa8"
"""

import argparse
from huggingface_hub import snapshot_download
from zipfile import ZipFile


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()
    dataset = args.dataset

    try:
        data_path = snapshot_download(
            repo_id=f"{dataset}",
            repo_type="dataset",
            cache_dir="Datasets")

        with ZipFile(f"{data_path}/data.zip", 'r') as zip:
            zip.extractall(f"{data_path}/Dataset")

        print(f"Downloaded dataset to: {data_path}")

    except BaseException as e:
        print(f"Failed to download dataset: {e}")
