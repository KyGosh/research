import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, default=os.path.join("d:\\", "Research", "test_data"))
    args = parser.parse_args()
    return

if __name__ == '__main__':
    main()