# MIDAS
import midas.utils

import os
import torch
import argparse

from boosting_monocular_depth_pipe import BoostingMonocularDepthPipeline
from PIL import Image


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input_image_path", type=str, required=True, help="input files directory " "Images can be .png .jpg .tiff"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="result dir. result depth will be png." " vides are JMPG as avi"
    )

    # Check for required input
    arguments, _ = parser.parse_known_args()
    print(arguments)

    input_image = Image.open(arguments.input_image_path)

    boosting_monocular_depth_pipeline = BoostingMonocularDepthPipeline(
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        pix2pix_model_path="pix2pix/checkpoints/mergemodel/latest_net_G.pth",
        depth_estimator_model_path="res101.pth",
    )

    depth_prediction = boosting_monocular_depth_pipeline(input_image)

    input_file_name = os.path.basename(arguments.input_image_path)
    output_file_name = os.path.join(arguments.output_dir, input_file_name)
    midas.utils.write_depth(output_file_name, depth_prediction)
