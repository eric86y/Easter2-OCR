import os
import argparse
from OCR.Config import CHARSET, STACK_FILE

from OCR.Trainer import OCRTrainer
from OCR.Networks import EasterNetwork
from OCR.Encoder import TibetanWylieEncoder, TibetanStackEncoder
from OCR.Utils import shuffle_data, create_dir, build_data_paths, read_stack_file


STACK_LIST = read_stack_file(STACK_FILE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=False, default="Output")
    parser.add_argument("--epochs", type=int, required=False, default=64)
    parser.add_argument("--start_scheduler", type=int, required=False, default=48)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--encoder", choices=["TibetanWylie", "TibetanStack"], required=False, default="TibetanWylie")

    args = parser.parse_args()
    dataset = args.dataset
    output_dir = args.output
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    start_scheduler = int(args.start_scheduler)
    label_encoder = args.encoder

    assert os.path.isdir(dataset)
    image_paths, label_paths = build_data_paths(dataset)

    assert len(image_paths) == len(label_paths)
    image_paths, label_paths = shuffle_data(image_paths, label_paths)

    create_dir(output_dir)

    image_width = 3200
    image_height = 100
    batch_size = 32
    lbl_encoder = TibetanWylieEncoder(CHARSET) if label_encoder == "TibetanWylie" else TibetanStackEncoder(STACK_LIST)
    num_classes = lbl_encoder.num_classes

    network = EasterNetwork(num_classes=num_classes, image_width=image_width, image_height=image_height, mean_pooling=True)
    workers = 4

    ocr_trainer = OCRTrainer(
        network=network,
        label_encoder=lbl_encoder,
        workers=workers,
        image_width=image_width,
        image_height=image_height,
        batch_size=batch_size,
        output_dir=output_dir,
        preload_labels=True)

    ocr_trainer.init(image_paths, label_paths)

    num_epochs = epochs
    scheduler_start = start_scheduler

    ocr_trainer.train(epochs=num_epochs, scheduler_start=scheduler_start, check_cer=True, export_onnx=True, silent=True)
