import os
import cv2
import numpy as np
import dlib


def read_data(data_path, annotation_path, test_csv_path, image_size=(40, 40)):
    listd = os.listdir(annotation_path)
    images = []
    X = []
    Y = []
    intest = []
    for filename in listd:
        path = annotation_path + "/" + filename
        with open(path, "r") as f:
            file = f.read()
            file = file.replace(",", "\n")
            file = file.replace(" ", " ")
            file = file.split("\n")
            image_name = file[0]
            xx = file[1::2]
            xx.pop()
            x = np.array(xx).astype(np.float)
            y = np.array(file[2::2]).astype(np.float)
            image = cv2.imread(data_path + "/" + str(image_name) + ".jpg")
            r, g, b = cv2.split(image)
            image = np.stack([b, g, r], axis=2)

            # BBOX
            pointx = np.array([x[58], x[63], x[64], x[66], x[70], x[78], x[76], x[80]])
            pointy = np.array([y[58], y[63], y[64], y[66], y[70], y[78], y[76], y[80]])
            x_min = np.min(pointx)
            y_min = np.min(pointy)

            x_max = np.max(pointx)
            y_max = np.max(pointy)
            pad = 20

            # Cropping lip
            BBox_x1 = int(x_min) - pad
            BBox_x2 = int(x_max) + pad
            BBox_y1 = int(y_min) - pad
            BBox_y2 = int(y_max) + pad
            crop_image = image[BBox_y1:BBox_y2, BBox_x1:BBox_x2, :]
            n = 58
            m = 114
            x = x[n:m] - BBox_x1
            y = y[n:m] - BBox_y1
            # resizing
            scale_image = cv2.resize(crop_image, image_size)
            fy = float(crop_image.shape[0]) / image_size[1]
            fx = float(crop_image.shape[1]) / image_size[0]
            x1 = x / fx
            y1 = y / fy

            images.append(scale_image)
            X.append(x1)
            Y.append(y1)
            if image_name in open(test_csv_path, "r").read():
                intest.append(True)
            else:
                intest.append(False)

    return images, X, Y, intest


def read_data_afw(data_path, annotation_path, test_csv_path, image_size):
    listd = os.listdir(annotation_path)
    images = []
    X = []
    Y = []
    intest = []
    for filename in listd:
        path = annotation_path + "/" + filename
        with open(path, "r") as f:
            file = f.read()
            file = file.replace(",", "\n")
            file = file.replace(" ", " ")
            file = file.split("\n")
            image_name = file[0]
            xx = file[1::2]
            xx.pop()
            x = np.array(xx).astype(np.float)
            y = np.array(file[2::2]).astype(np.float)
            image = cv2.imread(data_path + "/" + str(image_name) + ".jpg")
            r, g, b = cv2.split(image)
            image = np.stack([b, g, r], axis=2)

            # BBOX
            pointx = np.array([x[58], x[63], x[64], x[66], x[70], x[78], x[76], x[80]])
            pointy = np.array([y[58], y[63], y[64], y[66], y[70], y[78], y[76], y[80]])
            x_min = np.min(pointx)
            y_min = np.min(pointy)

            x_max = np.max(pointx)
            y_max = np.max(pointy)
            pad = 20

            # Cropping lip
            BBox_x1 = int(x_min) - pad
            BBox_x2 = int(x_max) + pad
            BBox_y1 = int(y_min) - pad
            BBox_y2 = int(y_max) + pad
            crop_image = image[BBox_y1:BBox_y2, BBox_x1:BBox_x2, :]
            #             n = 58
            #             m = 114
            #             x = x[n:m]-BBox_x1
            #             y = y[n:m]-BBox_y1
            pointx = pointx - BBox_x1
            pointy = pointy - BBox_y1
            # resizing
            scale_image = cv2.resize(crop_image, image_size)
            fy = float(crop_image.shape[0]) / image_size[1]
            fx = float(crop_image.shape[1]) / image_size[0]
            #             x1 = x/fx
            #             y1 = y/fy
            pointx = pointx / fx
            pointy = pointy / fy

            images.append(scale_image)
            X.append(pointx)
            Y.append(pointy)
            #             X.append(x1)
            #             Y.append(y1)
            if image_name in open(test_csv_path, "r").read():
                intest.append(True)
            else:
                intest.append(False)

    return images, X, Y, intest


def mean_X(images):
    zer = np.zeros([40, 40, 3])
    for i in range(0, len(images)):
        zer = zer + images[i]
    mean = zer / len(images)
    return mean


def normalize(images, m, std):
    images = images - m
    images = images / (std + 2e-10)
    return images


def normalizep(points, a=40, b=0.5):
    points = points / a
    points = points - b
    return points


def split_train_val(images, x, y, intest):
    # train test split
    test_images = []
    train_images = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(0, len(images)):

        if intest[i] == True:
            test_images.append(images[i])
            x_test.append(x[i])
            y_test.append(y[i])
        else:
            train_images.append(images[i])
            x_train.append(x[i])
            y_train.append(y[i])
    print("Number of training images: ", len(train_images))
    print("Number of testing images: ", len(test_images))

    train_points = np.concatenate([np.array(x_train), np.array(y_train)], axis=1)
    test_points = np.concatenate([np.array(x_test), np.array(y_test)], axis=1)

    print("Shape of training data :", np.array(x_train).shape)

    return train_images, test_images, train_points, test_points


def std_X(train_images, mean_image):
    zer = np.zeros([40, 40, 3])
    for i in range(0, len(train_images)):
        s = train_images[i] - mean_image.astype(int)
        zer = zer + s ** 2
    std = zer / len(train_images)
    std_image = np.sqrt(std)
    return std_image


def crop_mouth_with_dlib(image, shape_predictor_path):
    lips = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print(
            "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()
            )
        )
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # crop the mouth by  outer most points.
        xmouthpoints = [shape.part(x).x for x in range(48, 67)]
        ymouthpoints = [shape.part(x).y for x in range(48, 67)]
        maxx = max(xmouthpoints)
        minx = min(xmouthpoints)
        maxy = max(ymouthpoints)
        miny = min(ymouthpoints)

        # to show the mouth properly pad both sides
        pad = 20
        crop_image = image[miny - pad : maxy + pad, minx - pad : maxx + pad]
        lips.append(crop_image)
    return lips