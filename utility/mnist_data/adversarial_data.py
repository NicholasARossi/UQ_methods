from extra_keras_datasets import emnist
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Model


class MNIST_EMNIST:

    def __init__(self)->None:

        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        (self.X_holdout,self.y_holdout), (_,_) = emnist.load_data()


        # reshaping data
        self.X_train = self.X_train.reshape((self.X_train.shape[0], 28, 28, 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 28, 28, 1))
        self.X_holdout = self.X_holdout.reshape((self.X_holdout.shape[0], 28, 28, 1))

        self.prep_pixels()

        # one hot encode target values
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        self.y_holdout = to_categorical(self.y_holdout)
        self.train_keys = self.revere_categorical(self.y_train)
        self.holdout_keys = self.revere_categorical(self.y_holdout)
        self.adversarial_labels=self.holdout_keys>9

    @staticmethod
    def revere_categorical(array):
        labels=[]
        for a in array:
            labels.append(np.argmax(a, axis=None, out=None))
        return np.array(labels)

    def prep_pixels(self):
        # convert from integers to floats
        self.X_train = self.X_train.astype('float32')/ 255.0
        self.X_test = self.X_test.astype('float32')/ 255.0
        self.X_holdout = self.X_holdout.astype('float32')/ 255.0




def compute_dropout_uncertainty(model,X):

    # f = K.function([model.layers[0].input, backend.symbolic_learning_phase()],[model.layers[-1].output])

    partial_model = Model(model.inputs, model.layers[-1].output)
    result = []
    n_iter = 10
    for i in range(n_iter):
        result.append(partial_model(X,training=True))

    result = np.array(result)

    prediction_mean = result.mean(axis=0)
    prediction_std = result.std(axis=0)
    return prediction_mean,prediction_std,result