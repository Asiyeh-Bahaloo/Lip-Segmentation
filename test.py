import os
import argparse
from core.model import CasacadeSegmentor
import core.visualization as vis
from core.data import *
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for training the model")
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/image",
        help="Path to the test image",
        required=True,
    )
    parser.add_argument(
        "--mean_image_path",
        type=str,
        default="data/image",
        help="Path to the training images mean",
        required=True,
    )
    parser.add_argument(
        "--std_image_path",
        type=str,
        default="data/image",
        help="Path to the training images std",
        required=True,
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results",
        help="Path for saving the results",
        required=False,
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        help="Path to the directory containing all of the model's weights for loading",
        required=True,
    )
    parser.add_argument(
        "--shape_predictor_path",
        type=str,
        help="Path to dlib shape predictor dat file",
        required=True,
        default="./shape_predictor_68_face_landmarks.dat",
    )
    args = parser.parse_args()
    return args


def main():
    # get the parameters
    args = parse_arguments()
    w, h, channel = (40, 40, 3)

    # make output directory if it is not exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # read test,mean and std images
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_image = cv2.imread(args.mean_image_path)
    std_image = cv2.imread(args.std_image_path)
    if image is None or mean_image is None or std_image is None:
        print("Some of the pathes are wrong!!!")
        return

    # resize all images
    lips = crop_mouth_with_dlib(image, shape_predictor_path=args.shape_predictor_path)
    image = cv2.resize(lips[0], (w, h))
    mean_image = cv2.resize(mean_image, (w, h))
    std_image = cv2.resize(std_image, (w, h))

    # nomalize the test image
    norm_image = normalize(image, mean_image, std_image)

    # create model
    model = CasacadeSegmentor(input_shape=(w, h, channel), num_output=112, K=10)
    # load weights into the model
    model.load_weights(args.weights_path)

    # predict lip area
    result = model.predict(norm_image)
    vis.show_result(
        image,
        result[0:56],
        result[56:112],
        out_path=args.results_path + "/predict.jpg",
    )

    # show the lip segmentation
    vis.show_lip_segment(
        image,
        est_x=result[0:56],
        est_y=result[56:112],
        out_path=args.results_path + "/predict_segmentation.jpg",
    )
    print(
        "Segmentation result saved in: ",
        args.results_path + "/predict_segmentation.jpg",
    )


if __name__ == "__main__":
    main()
