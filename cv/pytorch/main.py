import argparse
import importlib
import numpy as np
import torch
import pytorch_lightning as pl


def _import_class(module_and_class_name: str) -> type:
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_class", type=str, default="FaceLandmarksDataset")
    
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"custom_datasets.data.{temp_args.data_class}")
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"custom_datasets.data.{args.data_class}")
    data = data_class(args)
    print(dir(data))

if __name__ == "__main__":
    main()
