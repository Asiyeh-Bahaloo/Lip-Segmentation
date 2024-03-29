import os
import argparse
from tabnanny import verbose
from core.model import CasacadeSegmentor
import core.visualization as vis
from core.data import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for training the model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/data/image",
        help="Path to the images folder",
        required=False,
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="Data/data/annotation",
        help="Path to the annotation folder",
        required=False,
    )
    parser.add_argument(
        "--test_csv_path",
        type=str,
        default="Data/data/testnames.txt",
        help="Path to the test idxs.",
        required=False,
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./output/",
        help="Path to save results",
        required=False,
    )
    parser.add_argument(
        "--learning_rate",
        dest="lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        required=False,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mean_squared_error",
        required=False,
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        help="Path to load model's weights",
        required=False,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of folder containing the saved model",
        default="model",
    )
    parser.add_argument(
        "--head_epochs",
        type=int,
        help="number of finetuning epochs for each cluster",
        default=100,
    )

    args = parser.parse_args()
    return args


def main():
    # get the parameters
    args = parse_arguments()
    input_shape = (40, 40, 3)
    model_path = os.path.join(args.results_path, args.model_name)

    # make output directory if it is not exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # read the training data
    images, x, y, intest = read_data(
        args.data_path, args.annotations_path, args.test_csv_path, input_shape[0:2]
    )

    # split the train and test data
    train_images, test_images, train_points, test_points = split_train_val(
        images, x, y, intest
    )

    # compute mean image
    mean_image = mean_X(images)

    # compute STD image
    std_image = std_X(train_images, mean_image)

    # nomalize data based on mean and STD images
    norm_train_images = normalize(train_images, mean_image, std_image)
    norm_test_images = normalize(test_images, mean_image, std_image)
    norm_train_points = normalizep(train_points)
    norm_test_points = normalizep(test_points)

    # create an instance of the model
    model = CasacadeSegmentor(input_shape=input_shape, num_output=112, K=10)

    # train the model with given parameters
    seq_history, heads_history = model.fit(
        norm_train_images,
        norm_train_points,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        loss=args.loss,
        metrics=["accuracy"],
        optimizer="RMSprop",
        verbose=1,
        head_epochs=args.head_epochs,
    )
    # save the trained model
    model.save(model_path)

    # evaluate the model performance
    print("***********Evaluating on train set*************")
    model.evaluate(
        norm_train_images, norm_train_points, metrics=["accuracy"], loss=args.loss
    )
    print("**********Evaluating on test set**************")
    model.evaluate(
        norm_test_images, norm_test_points, metrics=["accuracy"], loss=args.loss
    )

    # sample data visualization
    I = 16
    result = model.predict(norm_train_images[I, :, :, :])
    vis.show_result_gt(
        image=train_images[I],
        est_x=result[0:56],
        est_y=result[56:112],
        gt_x=train_points[I, 0:56],
        gt_y=train_points[I, 56:112],
        out_path=os.path.join(args.results_path, "lip_points_train.jpg"),
    )
    # I = 6
    # result = model.batch_predict(test_images[0:20, :, :, :])
    # vis.show_result_gt(
    #     image=test_images[I, :, :, :],
    #     est_x=result[I, 0:56],
    #     est_y=result[I, 56:112],
    #     gt_x=test_images[I, 0:56],
    #     gt_y=test_images[I, 56:112],
    #     out_path=args.results_path + "/result_test_6.jpg",
    # )
    vis.show_lip_segment(
        test_images[I],
        est_x=result[0:56],
        est_y=result[56:112],
        out_path=os.path.join(args.results_path, "lip_segmentation_test.jpg"),
    )
    vis.show_loss(
        seq_history, out_path=os.path.join(args.results_path, "backbone_loss.jpg")
    )
    vis.show_accuracy(
        seq_history, out_path=os.path.join(args.results_path, "backbone_accuracy.jpg")
    )
    vis.show_gmm_means(
        model.get_gmm_means(size=(12, 16, 3)),
        K=10,
        out_path=os.path.join(args.results_path, "gmm_mean"),
    )
    cnt = 0
    for history_head in heads_history:
        cnt = cnt + 1
        vis.show_loss(
            history_head,
            out_path=os.path.join(args.results_path, "head_loss" + str(cnt) + ".jpg"),
        )
        vis.show_accuracy(
            history_head,
            out_path=os.path.join(
                args.results_path, "head_accuracy" + str(cnt) + ".jpg"
            ),
        )


if __name__ == "__main__":
    main()