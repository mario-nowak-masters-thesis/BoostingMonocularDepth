from operator import getitem
from torchvision.transforms import Compose
from torchvision.transforms import transforms

# OUR
from utils import ImageAndPatches, ImageDataset, Images, generate_mask, getGF_fromintegral, calculate_processing_resolution, rgb2gray,\
    apply_grid_patch

# MIDAS
import midas.utils
from midas.models.midas_net import MidasNet
from midas.models.transforms import Resize, NormalizeImage, PrepareForNet

#AdelaiDepth
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import strip_prefix_if_present

# PIX2PIX : MERGE NET
from pix2pix.options.test_options import TestOptions
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

import time
import os
import torch
import cv2
import numpy as np
import numpy.typing as npt
import argparse
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from boosting_monocular_depth_pipe import BoostingMonocularDepthPipeline
from PIL import Image


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_image_path', type=str, required=True, help='input files directory '
                                                                    'Images can be .png .jpg .tiff')
    parser.add_argument('--output_dir', type=str, required=True, help='result dir. result depth will be png.'
                                                                      ' vides are JMPG as avi')


    # Check for required input
    arguments, _ = parser.parse_known_args()
    print(arguments)

    input_image = Image.open(arguments.input_image_path)

    boosting_monocular_depth_pipeline = BoostingMonocularDepthPipeline(
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        pix2pix_model_path="./pix2pix/checkpoints/mergemodel",
        leres_model_path="res101.pth"
    )

    depth_prediction = boosting_monocular_depth_pipeline(input_image)

    input_file_name = os.path.basename(arguments.input_image_path)
    output_file_name = os.path.join(arguments.output_dir, input_file_name)
    midas.utils.write_depth(output_file_name, depth_prediction)
