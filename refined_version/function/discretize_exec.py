import argparse
import json
import os.path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=os.path.join("d:\\", "Project", "Research", "cfg"))
    parser.add_argument('--input_dir', type=str, default=os.path.join("d:\\", "Project", "Research", "origin_data", "mouse_data"))
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = json.load(f)



if __name__ == '__main__':
    main()