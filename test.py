import os
import argparse
from model import CasacadeSegmentor
import visualization as vis
from data import *
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for training the model")
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/image",
        help="Path to the image",
        required=True,
    )
    parser.add_argument(
        "--mean_image_path",
        type=str,
        default="data/image",
        help="Path to the images mean",
        required=True,
    )
    parser.add_argument(
        "--std_image_path",
        type=str,
        default="data/image",
        help="Path to the images std",
        required=True,
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results",
        help="Path to save results",
        required=True,
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        help="Path to load model's weights",
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    # Parameters
    args = parse_arguments()
    input_shape = (40, 40, 3)

    # Make Directory
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # Read Data
    image = cv2.imread(args.image_path)
    mean_image = cv2.imread(args.mean_image_path)
    std_image = cv2.imread(args.std_image_path)

    # nomalize data
    image = normalize(image, mean_image, std_image)

    # create model
    model = CasacadeSegmentor(input_shape=input_shape, num_output=112, K=10)
    # load
    history_seq, history_heads = model.load(args.weights_path)

    # predict
    result = model.predict(image)
    vis.show_result(
        image,
        result[0:56],
        result[56:112],
        out_path=args.results_path + "/predict.jpg",
    )

    vis.show_lip_segment(
        image,
        est_x=result[0:56],
        est_y=result[56:112],
        nn=15,
        out_path=args.results_path + "/predict_segmentation.jpg",
    )
