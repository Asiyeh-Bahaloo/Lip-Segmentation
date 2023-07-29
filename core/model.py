import os, sys
import cv2
import numpy as np
from pickle import load, dump


from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

from keras.optimizers import SGD

from keras import initializers
from keras import layers
from sklearn.mixture import GaussianMixture
from keras.models import Model
from keras.models import clone_model
from keras.layers import Input

ROOT = ""
DATA_PATH = os.path.join(ROOT, "data/image")
ANNOTATION_PATH = os.path.join(ROOT, "data/annotation")
CSV_TEST = os.path.join(ROOT, "data", "testnames.txt")
CSV_TRAIN = os.path.join(ROOT, "data", "trainnames.txt")


class CasacadeSegmentor:
    def __init__(self, input_shape=(40, 40, 3), num_output=112, K=10):
        self.num_output = num_output
        self.input_shape = input_shape
        self.K = K
        self.model_seq, self.model_gmm = self.build(
            self.num_output, self.input_shape, self.K
        )
        self.heads = self.build_heads(self.model_seq, self.K)

    def build(self, num_output, input_shape, K, covariance_type="diag"):
        return self.build_sequencial(num_output, input_shape), self.build_gmm(
            K, covariance_type=covariance_type
        )

    def build_gmm(self, K, covariance_type):
        gmm = GaussianMixture(n_components=K, covariance_type=covariance_type)
        return gmm

    def build_heads(self, model_seq, K):
        heads = []
        for i in range(K):
            Tcnn = Sequential()
            for layer in model_seq.layers[8:10]:
                Tcnn.add(layer)
            heads.append(Tcnn)
        return heads

    # def init_weights_heads(self):
    #     for head in self.heads:
    #         head.layers[0].set_weights(self.model_seq.layers[8].get_weights())
    #         head.layers[1].set_weights(self.model_seq.layers[9].get_weights())

    def build_sequencial(self, num_output, input_shape):

        # Model Layers
        model = Sequential()
        model.add(
            Convolution2D(
                filters=16,
                kernel_size=5,
                strides=1,
                padding="same",
                activation="tanh",
                data_format="channels_last",
                input_shape=input_shape,
                kernel_initializer=initializers.glorot_normal(seed=None),
            )
        )  # xavier
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(
            Convolution2D(
                filters=48,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="tanh",
                data_format="channels_last",
                kernel_initializer=initializers.glorot_normal(seed=None),
            )
        )
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(
            Convolution2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="tanh",
                data_format="channels_last",
                kernel_initializer=initializers.glorot_normal(seed=None),
            )
        )
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(
            Convolution2D(
                filters=64,
                kernel_size=2,
                strides=1,
                padding="valid",
                activation="tanh",
                data_format="channels_last",
                kernel_initializer=initializers.glorot_normal(seed=None),
            )
        )
        model.add(Flatten())
        model.add(
            Dense(
                1024,
                activation="tanh",
                kernel_initializer=initializers.glorot_normal(seed=None),
            )
        )
        model.add(
            Dense(
                num_output,
                activation="tanh",
                kernel_initializer=initializers.glorot_normal(seed=None),
            )
        )
        return model

    def fit(
        self,
        X,
        Y,
        epochs=10000,
        batch_size=16,
        validation_split=0.15,
        loss="mean_squared_error",
        metrics=["accuracy"],
        optimizer="RMSprop",
        callbacks=[],
        head_epochs=1,
        **kwargs,
    ):
        # 1) Train the sequential part
        print("************ training sequential part ************")
        history_seq = self.train_sequential(
            X,
            Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            callbacks=callbacks,
            **kwargs,
        )
        # 2) Train GMM for clustering the images
        print("************ training GMM for clustering the images ************")
        features = self.intermediateFeat(X)
        self.model_gmm = self.train_gmm(features)
        labels = self.model_gmm.predict(features)
        cli_idxs = self.get_cli_idxs(labels)
        # 3) Fine Tune the heads with related cluster
        print("************ fine tuning the heads with related cluster ************")
        self.heads, head_history = self.build_and_train_heads(
            features,
            Y,
            cli_idxs,
            epochs=head_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            loss=loss,
            metrics=metrics,
            lr=0.001,
            momentum=0.99,
        )

        return history_seq.history, head_history

    def train_sequential(
        self,
        X,
        Y,
        epochs=10000,
        batch_size=16,
        validation_split=0.15,
        loss="mean_squared_error",
        metrics=["accuracy"],
        optimizer="RMSprop",
        callbacks=[],
        **kwargs,
    ):
        self.model_seq.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        history = self.model_seq.fit(
            np.array(X),
            np.array(Y),
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            **kwargs,
        )

        return history

    def train_gmm(self, X):
        X_gmm = np.array(X).reshape(len(X), -1)
        return self.model_gmm.fit(X_gmm)

    def build_and_train_heads(
        self,
        X,
        Y,
        cl_idxs,
        epochs=100,
        batch_size=16,
        validation_split=0.15,
        loss="mean_squared_error",
        metrics=["accuracy"],
        lr=0.001,
        momentum=0.99,
        callbacks=[],
        **kwargs,
    ):
        self.heads = []
        histories = []

        for i in range(self.K):
            Tcnn = Sequential()
            for layer in self.model_seq.layers[8:10]:
                Tcnn.add(layer)
            Tcnn.layers[0].set_weights(self.model_seq.layers[8].get_weights())
            Tcnn.layers[1].set_weights(self.model_seq.layers[9].get_weights())
            self.heads.append(Tcnn)

            self.heads[i].compile(
                optimizer=SGD(lr=lr, momentum=momentum),
                loss=loss,
                metrics=metrics,
            )
            if len(X[cl_idxs[i]]) < batch_size:
                batch_size = len(X[cl_idxs[i]])
                history = self.heads[i].fit(
                    X[cl_idxs[i]],
                    Y[cl_idxs[i]],
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    validation_split=0,
                    callbacks=callbacks,
                    **kwargs,
                )
            else:
                history = self.heads[i].fit(
                    X[cl_idxs[i]],
                    Y[cl_idxs[i]],
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    **kwargs,
                )
            histories.append(history.history)

        return self.heads, histories

    def get_cli_idxs(self, labels):
        cl_idxs = [[] for i in range(self.K)]
        for i, l in enumerate(labels):
            cl_idxs[l].append(i)
        print("************ Each Cluster population: ************")
        for j in range(self.K):
            print("cluster ", j, ": ", len(cl_idxs[j]))
        return cl_idxs

    def get_gmm_means(self, size):
        return self.model_gmm.means_.reshape(self.K, size[0], size[1], size[2])

    def evaluate(self, X, Y, metrics, loss, **kwargs):
        self.evaluate_sequential(X, Y, metrics, loss, **kwargs)
        self.evaluate_heads(X, Y, metrics, loss, **kwargs)
        return 0

    def evaluate_sequential(self, X, Y, metrics, loss, optimizer="RMSprop", **kwargs):
        self.model_seq.model.compile(optimizer=optimizer, metrics=metrics, loss=loss)
        (loss, accuracy) = self.model_seq.evaluate(X, Y, batch_size=32, **kwargs)

        print("loss : {}".format(loss))
        print("accuracy : {}".format(accuracy))

        return (loss, accuracy)

    def evaluate_heads(
        self, X, Y, metrics, loss, optimizer=SGD(lr=0.001, momentum=0.99), **kwargs
    ):
        feats = self.intermediateFeat(X)
        clusters = self.model_gmm.predict(feats.reshape(len(feats), -1))
        cnt = 0
        for head in self.heads:
            head.compile(optimizer=optimizer, metrics=metrics, loss=loss)
            idxs = np.argwhere(clusters == cnt)
            cluster_feat = feats[idxs].squeeze()
            cluster_Y = Y[idxs].squeeze()
            if len(idxs) == 1:
                print("len is one ", cnt)
                cluster_feat = np.expand_dims(cluster_feat, axis=0)
                cluster_Y = np.expand_dims(cluster_Y, axis=0)
            if len(idxs) != 0:
                (loss_v, accuracy) = head.evaluate(
                    cluster_feat,
                    cluster_Y,
                    batch_size=32,
                    **kwargs,
                )
                print("loss : {}".format(loss_v))
                print("accuracy : {}".format(accuracy))
                cnt += 1

    def predict(self, X, a=40, b=0.5):
        X = np.expand_dims(X, 0)
        feat = self.intermediateFeat(X)
        print("feat", feat.shape)
        label = self.model_gmm.predict(feat.reshape(len(feat), -1))[0]
        m = self.heads[label]
        f = feat[0, :]
        result = m.predict(
            np.array(
                [
                    f,
                ]
            )
        )
        pre = np.array(result).reshape(-1, 112).squeeze()
        return (pre + b) * a

    def batch_predict(self, X, a=40, b=0.5):
        result = [[] for i in range(len(X))]
        feat = self.intermediateFeat(X)
        labels = self.model_gmm.predict(feat.reshape(len(feat), -1))
        for i in range(len(result)):
            f = feat[i, :]
            m = self.heads[labels[i]]
            result[i] = m.predict(
                np.array(
                    [
                        f,
                    ]
                )
            )
        pre = np.array(result).reshape(-1, 112)
        return (pre + b) * a

    def intermediateFeat(self, X):
        extract = Model(self.model_seq.inputs, self.model_seq.layers[7].output)
        features = extract.predict(X)
        return features

    def save(self, folder_path, **kwargs):
        print("saving the model in ", folder_path)
        self.model_seq.save(folder_path + "/seq.h5", **kwargs)
        self.model_seq.save(folder_path + "/seq_gaph.pb", **kwargs)
        np.save(
            folder_path + "gmm_weights", self.model_gmm.weights_, allow_pickle=False
        )
        np.save(folder_path + "gmm_means", self.model_gmm.means_, allow_pickle=False)
        np.save(
            folder_path + "gmm_covariances",
            self.model_gmm.covariances_,
            allow_pickle=False,
        )
        for i in range(self.K):
            self.heads[i].save(folder_path + "/head" + str(i) + ".h5")
        return True

    def load_weights(self, folder_path, **kwargs):
        print("loading the model from ", folder_path)
        self.model_seq.load_weights(folder_path + "/seq.h5", **kwargs)
        self.model_gmm.weights_ = np.load(folder_path + "/gmm_weights.npy")
        self.model_gmm.means_ = np.load(folder_path + "/gmm_means.npy")
        self.model_gmm.covariances_ = np.load(folder_path + "/gmm_covariances.npy")
        for i in range(self.K):
            self.heads[i].load_weights(folder_path + "/head" + str(i) + ".h5")
        return True

    def summary(self):
        self.model.summary()
