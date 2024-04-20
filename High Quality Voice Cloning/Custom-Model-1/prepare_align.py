import argparse

import yaml

from preprocessor import libritts, youshan


def main(config):
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    if 'youshan' in config['dataset']:
        youshan.prepare_align(config)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
