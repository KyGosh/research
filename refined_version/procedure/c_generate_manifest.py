# generate the training manifest file, freezing the dataset used for model training
import argparse
import os

import pandas as pd
DATA_TYPE=["mouse", "keyboard", "combined"]

def generate_manifest(pt_idx: str, player: str, output_file, type):

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-idx", type=str, default=os.path.join("d:\\", "Project", "Research", "pt_data"))
    parser.add_argument("--player", type=str, default="apEX")
    parser.add_argument("--manifest-file", type=str, default=os.path.join("d:\\", "Project", "Research", "manifest_file"))
    args = parser.parse_args()
    output_file = os.path.join(args.manifest_file, f"{args.player}.json")
    for type in DATA_TYPE:
        generate_manifest(args.pt_idx, args.player, output_file, type)

if __name__ == '__main__':
    main()