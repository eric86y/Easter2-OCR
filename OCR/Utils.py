import os
import re
import cv2
import json
import torch
import logging
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
import numpy.typing as npt
from natsort import natsorted
from botok import normalize_unicode
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

from OCR.Data import OCRModelConfig


def create_dir(dir_path: str) -> None:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created output directory: {dir_path}")
    except OSError as e:
        logging.error(f"Failed to create directory: {e}")


def get_filename(file_path: str) -> str:
    base_name = os.path.basename(file_path) 
    file_name = os.path.splitext(base_name)[0]
    return file_name


def read_stack_file(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        stacks = f.readlines()
        stacks = [x.replace("\n", "") for x in stacks]
        stacks = [x.replace("\t", "") for x in stacks]

        return stacks


def read_ocr_model_config(config_file: str):
    model_dir = os.path.dirname(config_file)

    try:
        with open(config_file, encoding="utf-8") as file:
            json_content = json.load(file)
    except FileNotFoundError: 
        raise FileNotFoundError(f"Configuration file {config_file} not found.")

    except json.JSONDecodeError: 
        raise ValueError(f"Error decoding JSON from {config_file}")

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    architecture = json_content["architecture"]
    input_width = json_content["input_width"]
    input_height = json_content["input_height"]
    input_layer = json_content["input_layer"]
    output_layer = json_content["output_layer"]
    encoder = json_content["encoder"]
    squeeze_channel_dim = (
        True if json_content["squeeze_channel_dim"] == "yes" else False
    )
    swap_hw = True if json_content["swap_hw"] == "yes" else False
    characters = json_content["charset"]
    add_blank = True if json_content["add_blank"] == "yes" else False

    config = OCRModelConfig(
        onnx_model_file,
        architecture,
        input_width,
        input_height,
        input_layer,
        output_layer,
        squeeze_channel_dim,
        swap_hw,
        encoder=encoder,
        charset=characters,
        add_blank=add_blank
    )

    return config


def read_distribution(distribution_file: str):   
    try:
        with open(distribution_file, "r", encoding="utf-8") as f:
            content = f.read()
            content = json.loads(content)

            if "train" in content and "validation" in content and "test" in content:
                train_samples = content["train"]
                valid_samples = content["validation"]
                test_samples = content["test"]

                return train_samples, valid_samples, test_samples
            else:
                logging.error(
                    "Data distribution is missing the required keys 'train' and 'validation' and 'test'."
                )
                return None, None, None
    except FileNotFoundError:
        logging.error(f"File not found: {distribution_file}")
        return None, None, None 
    
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {distribution_file}")
        return None, None, None


def build_data_paths(data_root: str) -> Tuple[List[str], List[str]]:
    images = natsorted(glob(f"{data_root}/lines/*.jpg"))
    labels = natsorted(glob(f"{data_root}/transcriptions/*.txt"))

    return images, labels


def build_distribution_paths(
    data_path: str, samples: List[str]
) -> Tuple[List[str], List[str]]:
    images = []
    labels = []

    for sample in samples:
        image = f"{data_path}/lines/{sample}.jpg"
        label = f"{data_path}/transcriptions/{sample}.txt"

        if os.path.isfile(image) and os.path.isfile(label):
            images.append(image)
            labels.append(label)

        else:
            logging.warning(f"Warning: image-label pair not found: {sample}")

    return images, labels


def assemble_data_paths(data_root: Path) -> Tuple[List[Path], List[Path]]:
    assert os.path.isdir(data_root)

    image_paths = []
    label_paths = []

    for ds_dir in data_root.iterdir():
        images = natsorted(glob(f"{ds_dir}/lines/*.jpg"))
        labels = natsorted(glob(f"{ds_dir}/transcriptions/*.txt"))

        if len(images) > 0 and len(labels) > 0 and len(images) == (len(labels)):
            logging.warning(f"{ds_dir.name} => Images: {len(images)}, Labels:{len(labels)}")
            image_paths.extend(images)
            label_paths.extend(labels)

        else:
            logging.warning(f"Mismatch or empty images/labels in {ds_dir.name}")
            return [], []

    return image_paths, label_paths


def build_distribution_from_directory(data_dir: str) -> Dict:
    data_path = Path(data_dir)

    all_train_images = []
    all_valid_images = []
    all_test_images = []

    all_train_labels = []
    all_valid_labels = []
    all_test_labels = []

    for sub_dir in data_path.iterdir():
        if sub_dir.name == "Output":
            continue

        _images = natsorted(glob(f"{sub_dir}/lines/*.jpg"))
        _labels = natsorted(glob(f"{sub_dir}/transcriptions/*.txt"))
        print(f"{sub_dir.name} => Images: {len(_images)}, Labels: {len(_labels)}")

        if len(_images) != len(_labels):

            image_list = list(map(get_filename, _images))
            labels_list = list(map(get_filename, _labels))

            shared_list = list(set(image_list) & set(labels_list))

            shared_images = [f"{sub_dir}/lines/{x}.jpg" for x in shared_list]
            share_labels = [f"{sub_dir}/transcriptions/{x}.txt" for x in shared_list]

            _images = shared_images
            _labels = share_labels

        for _, (img, lbl) in tqdm(enumerate(zip(_images, _labels)), total=len(_images)):

            img_n = get_filename(img)
            lbl_n = get_filename(lbl)

            if not img_n == lbl_n:
                print(f"Warning: Label name mismatch: {img} => {lbl} ")

        _images, _labels = shuffle_data(_images, _labels)
        train_images, train_labels, val_images, val_labels, test_images, test_labels = (
            split_dataset(_images, _labels)
        )

        all_train_images.extend(train_images)
        all_train_labels.extend(train_labels)
        all_valid_images.extend(val_images)
        all_valid_labels.extend(val_labels)
        all_test_images.extend(test_images)
        all_test_labels.extend(test_labels)

        print(f"Train Images: {len(train_images)}, Train Labels: {len(train_labels)}")
        print(f"Val Images: {len(val_images)}, Val Labels: {len(val_labels)}")
        print(f"Test Images: {len(test_images)}, Test Labels: {len(test_labels)}")

        all_train_images, all_train_labels = shuffle_data(
            all_train_images, all_train_labels
        )
        all_valid_images, all_valid_labels = shuffle_data(
            all_valid_images, all_valid_labels
        )
        all_test_images, all_test_labels = shuffle_data(
            all_test_images, all_test_labels
        )

        validate_split(all_train_images, all_train_labels)
        validate_split(all_valid_images, all_valid_labels)
        validate_split(all_test_images, all_test_labels)

    save_distribution(
        all_train_images, all_valid_images, all_test_images, output_dir=data_dir
    )

    distribution = {}
    distribution["train_images"] = all_train_images
    distribution["train_labels"] = all_train_labels
    distribution["valid_images"] = all_valid_images
    distribution["valid_labels"] = all_valid_labels
    distribution["test_images"] = all_test_images
    distribution["test_labels"] = all_test_labels

    return distribution


def accumulate_distributions(data_path: str, datasets: List[str]):
    all_train_images = []
    all_train_labels = []

    all_val_images = []
    all_val_labels = []

    all_test_images = []
    all_test_labels = []

    for dataset in datasets:
        sub_dir = os.path.join(data_path, dataset)
        distr_file = f"{sub_dir}/data.distribution"

        assert os.path.isdir(sub_dir)
        assert os.path.isfile(distr_file)

        train_samples, valid_samples, test_samples = read_distribution(distr_file)

        train_images, train_labels = build_distribution_paths(sub_dir, train_samples)
        val_images, val_labels = build_distribution_paths(sub_dir, valid_samples)
        test_images, test_labels = build_distribution_paths(sub_dir, test_samples)

        all_train_images.extend(train_images)
        all_train_labels.extend(train_labels)

        all_val_images.extend(val_images)
        all_val_labels.extend(val_labels)

        all_test_images.extend(test_images)
        all_test_labels.extend(test_labels)

    all_train_images, all_train_labels = shuffle_data(
        all_train_images, all_train_labels
    )
    all_val_images, all_val_labels = shuffle_data(all_val_images, all_val_labels)
    all_test_images, all_test_labels = shuffle_data(all_test_images, all_test_labels)

    distribution = {}
    distribution["train_images"] = all_train_images
    distribution["train_labels"] = all_train_labels
    distribution["valid_images"] = all_val_images
    distribution["valid_labels"] = all_val_labels
    distribution["test_images"] = all_test_images
    distribution["test_labels"] = all_test_labels

    return distribution


def build_distribution_from_file(distribution_file: str, data_root: str) -> Dict:
    train_samples, valid_samples, test_samples = read_distribution(distribution_file)

    train_images, train_labels = build_distribution_paths(data_root, train_samples)
    val_images, val_labels = build_distribution_paths(data_root, valid_samples)
    test_images, test_labels = build_distribution_paths(data_root, test_samples)

    distribution = {}
    distribution["train_images"] = train_images
    distribution["train_labels"] = train_labels
    distribution["valid_images"] = val_images
    distribution["valid_labels"] = val_labels
    distribution["test_images"] = test_images
    distribution["test_labels"] = test_labels

    return distribution


def save_distribution(
    train_images: List[str],
    valid_images: List[str],
    test_images: List[str],
    output_dir: str,
):
    assert os.path.isdir(output_dir)

    out_file = os.path.join(output_dir, "data.distribution")

    distribution = {}
    train_data = []
    valid_data = []
    test_data = []

    for sample in train_images:
        sample_name = get_filename(sample)
        train_data.append(sample_name)

    for sample in valid_images:
        sample_name = get_filename(sample)
        valid_data.append(sample_name)

    for sample in test_images:
        sample_name = get_filename(sample)
        test_data.append(sample_name)

    distribution["train"] = train_data
    distribution["validation"] = valid_data
    distribution["test"] = test_data

    with open(out_file, "w", encoding="UTF-8") as f:
        json.dump(distribution, f, ensure_ascii=False, indent=1)

    print(f"Saved data distribution to: {out_file}")


def split_dataset(
    images: List[str],
    labels: List[str],
    train_val_split: float = 0.2,
    val_test_split: float = 0.5,
    seed: int = 42,
):
    train_images, valtest_images, train_labels, valtest_labels = train_test_split(
        images, labels, test_size=train_val_split, random_state=seed
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        valtest_images, valtest_labels, test_size=val_test_split, random_state=seed
    )

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def validate_split(images: List[str], labels: List[str]):
    for _, (img, lbl) in tqdm(enumerate(zip(images, labels)), total=len(images)):
        img_n = get_filename(img)
        lbl_n = get_filename(lbl)

        if not img_n == lbl_n:
            print(f"Mismatch: {img} vs. {lbl}")


def shuffle_data(images: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    c = list(zip(images, labels))
    random.shuffle(c)

    a, b = zip(*c)

    return list(a), list(b)


def binarize(
    image: npt.NDArray, adaptive: bool = True, block_size: int = 51, c: int = 13
) -> npt.NDArray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if adaptive:
        bw = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    else:
        _, bw = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

    bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return bw


def resize_to_height(image, target_height: int) -> Tuple[npt.NDArray, int]:
    ratio = target_height / image.shape[0]
    image = cv2.resize(image, (int(image.shape[1] * ratio), target_height))
    return image, ratio


def resize_to_width(image, target_width: int) -> Tuple[npt.NDArray, int]:
    ratio = target_width / image.shape[1]
    image = cv2.resize(image, (target_width, int(image.shape[0] * ratio)))
    return image, ratio


def resize(image: npt.NDArray, target_width: int, target_height: int) -> npt.NDArray:
    width_ratio = target_width / image.shape[1]
    height_ratio = target_height / image.shape[0]

    if width_ratio < height_ratio:  # maybe handle equality separately
        tmp_img, _ = resize_to_width(image, target_width)

    elif width_ratio > height_ratio:
        tmp_img, _ = resize_to_height(image, target_height)

    else:
        tmp_img, _ = resize_to_width(image, target_width)

    return cv2.resize(tmp_img, (target_width, target_height))


def pad_to_width(
    image: npt.NDArray, target_width: int, target_height: int, padding: str
) -> npt.NDArray:
    tmp_img, _ = resize_to_width(image, target_width)

    height = tmp_img.shape[0]
    middle = (target_height - tmp_img.shape[0]) // 2

    if padding == "white":
        upper_stack = np.ones(shape=(middle, target_width), dtype=np.uint8)
        lower_stack = np.ones(
            shape=(target_height - height - middle, target_width), dtype=np.uint8
        )

        upper_stack *= 255
        lower_stack *= 255
    else:
        upper_stack = np.zeros(shape=(middle, target_width), dtype=np.uint8)
        lower_stack = np.zeros(
            shape=(target_height - height - middle, target_width), dtype=np.uint8
        )

    out_img = np.vstack([upper_stack, tmp_img, lower_stack])

    return out_img


def pad_to_height(
    image: npt.NDArray[np.uint8], target_width: int, target_height: int, padding: str
) -> npt.NDArray[np.uint8]:
    """
    Pad an image to the target height while keeping the aspect ratio.
    Args:
        image: Input image as a numpy array.
        target_width: Desired width of the output image.
        target_height: Desired height of the output image.
        padding: "white" for padding the image with 255, otherwise the image will be padded with 0.
    Returns:
        Padded image as a numpy array.
    """
    tmp_img, _ = resize_to_height(image, target_height)

    width = tmp_img.shape[1]
    middle = (target_width - width) // 2

    padding_value = 255 if padding == "white" else 0
    left_stack = np.full((target_height, middle), padding_value, dtype=np.uint8)
    right_stack = np.full((target_height, target_width - width - middle), padding_value, dtype=np.uint8)

    out_img = np.hstack([left_stack, tmp_img, right_stack])

    return out_img

# Example usage
# Make sure resize_to_height is defined elsewhere in your code



def pad_ocr_line(
    image: npt.NDArray,
    target_width: int = 2000,
    target_height: int = 80,
    padding: str = "black",
) -> npt.NDArray:

    width_ratio = target_width / image.shape[1]
    height_ratio = target_height / image.shape[0]

    if width_ratio < height_ratio:
        out_img = pad_to_width(image, target_width, target_height, padding)

    elif width_ratio > height_ratio:
        out_img = pad_to_height(image, target_width, target_height, padding)
    else:
        out_img = pad_to_width(image, target_width, target_height, padding)

    return cv2.resize(
        out_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )


def resize_n_pad(
    image: npt.NDArray, target_width: int, target_height: int, padding: str
) -> npt.NDArray:
    """
    Resize and pad images to target dimensions.
    Args:
        - image: Input image as a numpy array.
        - target_width: Desired width of the output image.
        - target_height: Desired height of the output image.
        - padding: "white" for padding the image with 255, otherwise the image will be padded with 0.
    Returns:
        - Resized and padded image as a numpy array.
    """
    width_ratio = target_width / image.shape[1]
    height_ratio = target_height / image.shape[0]

    if width_ratio < height_ratio:
        tmp_img, _ = resize_to_width(image, target_width)
        padding_value = 255 if padding == "white" else 0
        v_stack = np.full((target_height - tmp_img.shape[0], tmp_img.shape[1]), padding_value, dtype=np.uint8)
        out_img = np.vstack([v_stack, tmp_img])

    elif width_ratio > height_ratio:
        tmp_img, _ = resize_to_height(image, target_height)
        padding_value = 255 if padding == "white" else 0
        h_stack = np.full((tmp_img.shape[0], target_width - tmp_img.shape[1]), padding_value, dtype=np.uint8)
        out_img = np.hstack([tmp_img, h_stack])
        
    else:
        tmp_img, _ = resize_to_width(image, target_width)
        padding_value = 255 if padding == "white" else 0
        v_stack = np.full((target_height - tmp_img.shape[0], tmp_img.shape[1]), padding_value, dtype=np.uint8)
        out_img = np.vstack([v_stack, tmp_img])

    return cv2.resize(out_img, (target_width, target_height))

# TODO: Improve this function using np.pad for a more elegant and potentially faster implementation



def preprocess_unicode(label: str, full_bracket_removal: bool = False) -> str:
    """
    Some preliminary clean-up rules for the Unicode text.
    - Note: () are just removed. This was valid in case of the Lhasa Kanjur.
    In other e-texts, a complete removal of the round and/or square brackets
    together with the enclosed text should be applied
    in order to remove interpolations, remarks or similar additions.
    In such cases set full_bracket_removal to True.
    """
    label = label.replace("\uf8f0", " ")
    label = label.replace("", "")
    label = label.replace("\xa0", "")
    label = label.replace("\x10", "")
    label = label.replace("\t", "")
    label = label.replace("\u200d", "")
    label = label.replace("\uf037", "")
    label = label.replace("\uf038", "")
    label = label.replace("༌", "་")  # replace triangle tsheg with regular

    if full_bracket_removal:
        label = re.sub(r"[\[(].*?[\])]", "", label)
    else:
        label = re.sub("[()]", "", label)
    return label


def preprocess_wylie_label(label: str) -> str:
    label = label.replace("༈", "!")
    label = label.replace("＠", "@")
    label = label.replace("।", "|")
    label = label.replace("༅", "#")
    label = label.replace("|", "/")  # TODO: let sb. verify this choice is ok
    label = label.replace("/ /", "/_/")
    label = label.replace("/ ", "/")

    return label


def postprocess_wylie_label(label: str) -> str:
    label = label.replace("\\u0f85", "&")
    label = label.replace("\\u0f09", "ä")
    label = label.replace("\\u0f13", "ö")
    label = label.replace("\\u0f12", "ü")
    label = label.replace("\\u0fd3", "@")
    label = label.replace("\\u0fd4", "#")
    label = label.replace("\\u0f00", "oM")
    label = label.replace("\\u0f7f", "}")
    label = label.replace("＠", "@")
    label = label.replace("।", "|")
    label = label.replace("*", " ")
    label = label.replace("  ", " ")
    label = label.replace("_", "")
    label = label.replace("[", "")
    label = label.replace("]", "")
    label = label.replace(" ", "§")  # specific encoding for the tsheg

    return label


def read_data(
    image_list: List,
    label_list: List,
    converter,
    min_label_length: int = 30,
    max_label_length: int = 320,
) -> Tuple[List[str], List[str]]:
    """
    Reads all labels into memory(!), filter labels for min_label_length and max_label_length.
    """
    labels = []
    images = []
    for image_path, label_path in tqdm(
        zip(image_list, label_list), total=len(label_list), desc="reading labels"
    ):
        f = open(label_path, "r", encoding="utf-8")
        label = f.readline()

        try:
            label = normalize_unicode(label)
            label = preprocess_unicode(label)
        except BaseException as e:
            print(f"Failed to preprocess unicode label: {label_path}, {e}")

        if min_label_length < len(label) < max_label_length:
            label = converter.toWylie(label)
            label = postprocess_wylie_label(label)

            if "\\u" not in label:  # filter out improperly converted unicode signs
                labels.append(label)
                images.append(image_path)

    return images, labels


def ctc_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
