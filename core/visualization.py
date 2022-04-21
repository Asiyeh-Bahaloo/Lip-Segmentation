import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


def show_data_sample(
    image,
    gt_x,
    gt_y,
    out_path=None,
):
    plt.scatter(gt_x, gt_y, s=2)
    plt.imshow(image)
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_image(image, out_path=None):
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
    nn=15,
    out_path=None,
):
    plt.scatter(est_x, est_y, s=2)
    nn = 15
    x_new = np.linspace(est_x[0:nn].min(), est_y[0:nn].max(), 50)
    f = interp1d(est_x[0:nn], est_y[56 : 56 + nn], kind="quadratic")
    y_smooth = f(x_new)
    plt.plot(x_new, y_smooth)
    plt.imshow(image)
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_loss(history, out_path=None):
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="center right")
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_accuracy(history, out_path=None):
    plt.plot(history["acc"])
    plt.plot(history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="center right")
    if out_path != None:
        plt.savefig(out_path)
    else:
        plt.show()


def show_gmm_means(gmm_means, K=10, out_path=None):
    for i in range(0, K):
        plt.imshow(gmm_means[i, :, :, 0:3])
        if out_path != None:
            plt.savefig(out_path)
        else:
            plt.show()
