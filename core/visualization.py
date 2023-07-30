import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import cv2


def show_data_sample(
    image,
    gt_x,
    gt_y,
    out_path=None,
):
    plt.clf()
    plt.scatter(gt_x, gt_y, s=2)
    plt.imshow(image)
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_image(image, out_path=None):
    plt.clf()
    plt.imshow(image.astype(int))
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_result_gt(
    image,
    est_x,
    est_y,
    gt_x,
    gt_y,
    out_path=None,
):
    plt.clf()
    plt.scatter(gt_x, gt_y, s=2)
    plt.scatter(est_x, est_y, s=2)
    plt.imshow(image)
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_result(
    image,
    est_x,
    est_y,
    out_path=None,
):
    plt.clf()
    plt.scatter(est_x, est_y, s=2)
    plt.imshow(image)
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_lip_segment(
    image,
    est_x,
    est_y,
    out_path=None,
):
    plt.clf()
    points = [(x, y) for x, y in zip(est_x, est_y)]
    contours = [np.array(points).reshape((-1, 1, 2)).astype(np.int32)]

    cv2.drawContours(image, contours, 0, (255, 0, 0), thickness=cv2.FILLED)
    plt.imshow(image)
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_loss(history, out_path=None):
    plt.clf()
    plt.plot(history["loss"])
    try:
        plt.plot(history["val_loss"])
    except:
        print("No validation data is available")
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="center right")
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_accuracy(history, out_path=None):
    plt.clf()
    plt.plot(history["accuracy"])
    try:
        plt.plot(history["val_accuracy"])
    except:
        print("No validation data is available")
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="center right")
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_gmm_means(gmm_means, K=10, out_path=None):
    plt.clf()
    for i in range(0, K):
        plt.imshow((gmm_means[i, :, :, 0:3] * 255).astype(np.uint8))
        if out_path != None:
            plt.savefig(out_path + str(i) + ".jpg")
        else:
            plt.show()
