from classifier import Classifier
from random import seed, shuffle
import numpy as np


class SVM_SGD(Classifier):
    """A SVM_SGD Classifier """
    def __init__(self, n_features):
        self.n_features = n_features
        self.learning_rate = 0.0001
        self.weight = np.zeros((1, n_features))
        self.iter = 10
        self.batch_size = 1
        self.cnt_error = 0
        self.cnt_train = 0
        self.interval = 1000

    def get_model(self):
        return model

    def set_model(self, model):
        self.model = model

    model = property(get_model, set_model)

    def train(self, train_x, train_y):
        # Calcualte the num of batches
        num_batches = len(train_x) / self.batch_size
        for i in range(self.iter):
            # shuffle(train_x)
            for j in range(num_batches):
                # Find the batch of the 
                batch_x = train_x[j * self.batch_size: (j + 1) * self.batch_size, :]
                batch_y = train_y[j * self.batch_size: (j + 1) * self.batch_size, :]
                self.train_sgd(batch_x, batch_y)
        

    def train_sgd(self, batch_x, batch_y):
        for i in range(len(batch_x)):
            # Use the hinge loss function to calculate error
            error = self.compute_hinge_error(batch_x[i, :], batch_y[i, :])
            self.cnt_train += 1
            if error > 0:
                self.cnt_error += 1
                self.correct(batch_x[i, :], batch_y[i, :])
            # self.status()

    def correct(self, batch_x, batch_y):
        """Correct the model. Pegasos algorithm """
        b = 1 - 1.0 / self.cnt_train
        self.scale_in_place(b);
        self.add_in_place(batch_x, batch_y)
        norm = np.sqrt(np.inner(self.weight, self.weight))
        scale = 1.0 / (np.sqrt(self.learning_rate) * norm)
        if scale < 1.0:
            self.scale_in_place(scale)

    def scale_in_place(self, b):
        # Change the weight vector in the SVM with w[i] = w[i] * b
        temp = np.multiply(self.weight, b)
        self.weight = temp

    def add_in_place(self, feature, label):
        b = label * 1.0 / (self.learning_rate * self.cnt_train) 
        temp = np.multiply(feature, b)
        t = np.add(self.weight, temp)
        self.weight = t

    def compute_hinge_error(self, feature, label):
        # Hinge loss funtion max(0, y(w*x))
        return max(0, 1 - label * np.inner(self.weight, feature))

    def status(self):
        if self.cnt_train % self.interval == 0:
            print("Step: " + repr(self.cnt_train))
            print("There are total " + repr(self.cnt_error) + " errors")
            print("accuracy is " + repr(1 - 1.0 * self.cnt_error / self.cnt_train))

    def classify(self, test_x, test_y):
        result = np.sign(np.inner(self.weight, test_x))
        correct = result == test_y.T
        accuracy = float(np.sum(correct)) / len(test_y)
        print("The accuracy of the sgd svm classifier is: %.3f%%" % (accuracy * 100))