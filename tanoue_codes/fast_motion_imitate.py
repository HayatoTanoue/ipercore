import os
import os.path as osp
import argparse
import platform
import time

from iPERCore.services.options.options_setup import setup
from iPERCore.services.run_imitator import run_imitator


def build_args():
    parser = argparse.ArgumentParser(
        description="Run motion imitation with optional pre-processing cache")

    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU ids to use")
    parser.add_argument("--image_size", type=int, default=512, help="input image size")
    parser.add_argument("--num_source", type=int, default=2, help="number of source images")
    parser.add_argument("--output_dir", type=str, default="./results", help="directory to save results")
    parser.add_argument(
        "--assets_dir", type=str, default="./assets",
        help="the assets directory containing configs and pre-trained checkpoints")
    parser.add_argument(
        "--model_id", type=str, default=f"model_{int(time.time())}",
        help="name of the model directory")
    parser.add_argument("--source_dir", type=str, required=True, help="directory of source images")
    parser.add_argument("--target_path", type=str, required=True, help="path to target dance video")
    parser.add_argument(
        "--preprocess_only", action="store_true",
        help="only preprocess inputs and exit")

    return parser.parse_known_args()


def main():
    args, extra_args = build_args()

    work_assets_dir = "./assets"
    if not os.path.exists(work_assets_dir):
        os.symlink(
            osp.abspath(args.assets_dir),
            osp.abspath(work_assets_dir),
            target_is_directory=(platform.system() == "Windows")
        )

    args.cfg_path = osp.join(work_assets_dir, "configs", "deploy.toml")

    args.src_path = f"path?={args.source_dir},name?={osp.basename(args.source_dir)}"
    args.ref_path = f"path?={args.target_path},name?={osp.basename(args.target_path)}"

    cfg = setup(args, extra_args)

    if args.preprocess_only:
        from iPERCore.services.preprocess import preprocess
        preprocess(cfg)
    else:
        run_imitator(cfg)


if __name__ == "__main__":
    main()