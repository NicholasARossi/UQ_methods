from extra_keras_datasets import emnist
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Model


class MNIST_EMNIST:

    def __init__(self,select_adversarials=True)->None:

        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        (self.X_adv_train,self.y_adv_train), (self.X_adv_test,self.y_adv_test) = emnist.load_data()


        # reshaping data
        self.X_train = self.featurize_input(self.X_train)
        self.X_test = self.featurize_input(self.X_test)
        self.X_adv_train = self.featurize_input(self.X_adv_train)
        self.X_adv_test = self.featurize_input(self.X_adv_test)


        # one hot encode target values
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        self.y_adv_train = to_categorical(self.y_adv_train)
        self.y_adv_test = to_categorical(self.y_adv_test)

        self.train_keys = self.revere_categorical(self.y_train)
        self.holdout_training_keys = self.revere_categorical(self.y_adv_train)
        self.holdout_testing_keys = self.revere_categorical(self.y_adv_test)

        # is it a number or a letter?
        self.adversarial_training_labels=self.holdout_training_keys>9
        self.adversarial_testing_labels=self.holdout_testing_keys>9


        if select_adversarials == True:
            select_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                            10,12,14,15,17,20,22,23,27,33]

            selection_mask = [lab in select_labels for lab in self.holdout_training_keys]
            self.X_adv_train = self.X_adv_train[selection_mask]
            self.y_adv_train = self.y_adv_train[selection_mask]
            self.adversarial_training_labels = self.adversarial_training_labels[selection_mask]
            self.holdout_training_keys = self.holdout_training_keys[selection_mask]


            selection_mask2=[lab in select_labels for lab in self.holdout_testing_keys]
            self.X_adv_test = self.X_adv_test[selection_mask2]
            self.y_adv_test=self.y_adv_test[selection_mask2]
            self.adversarial_testing_labels = self.adversarial_testing_labels[selection_mask2]
            self.holdout_testing_keys = self.holdout_testing_keys[selection_mask2]



    def get_taste_test(self):
        trainers = []
        for key in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            trainers.append(self.X_train[self.train_keys == key][0])

        adversarials = []
        for key in [10,12,14,15,17,20,22,23,27,33]:
            adversarials.append(self.X_adv_train[self.holdout_training_keys == key][0])

        return trainers,adversarials






    @staticmethod
    def revere_categorical(array:np.array)->np.array:
        labels=[]
        for a in array:
            labels.append(np.argmax(a, axis=None, out=None))
        return np.array(labels)

    @staticmethod
    def featurize_input(array:np.array)->np.array:
        # reshape
        array = array.reshape((array.shape[0], 28, 28, 1))
        # prep pixels // convert from integers to floats
        array = array.astype('float32')/ 255.0
        return array





def compute_dropout_uncertainty(model,X,n_iter = 10):

    # f = K.function([model.layers[0].input, backend.symbolic_learning_phase()],[model.layers[-1].output])

    partial_model = Model(model.inputs, model.layers[-1].output)
    result = []
    for i in range(n_iter):
        result.append(partial_model(X,training=True))

    result = np.array(result)

    prediction_mean = result.mean(axis=0)
    prediction_std = result.std(axis=0)
    return prediction_mean,prediction_std