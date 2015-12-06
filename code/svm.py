from classifier import Classifier
from kernel import Kernel
import cvxopt.solvers
import numpy as np
import random


class SVM(Classifier):
    """A SVM classifier """

    def __init__(self, label, featureDict, kernel = Kernel.linear, c = None):
        self.label = label # label dict of all the dataset
        self.featureDict = featureDict # feature dictionary for all the dataset
        self.kernel = kernel
        self.c = 0.6
        if self.c is not None:
            self.c = float(self.c)

    def get_model(self):
        return model

    def set_model(self, model):
        self.model = model

    model = property(get_model, set_model)

    def getSparseFeatures(self, training_set):
        num_instances = len(training_set)
        x = []
        y = np.zeros((num_instances, 1))

        for i in range(num_instances):
            sparse = []
            instance = training_set[i]
            y[i, 0] = self.label.get(instance.label) 

            for f in instance.features().split():
                sparse.append(self.featureDict.get(f))
            x.append(sparse)
        return mat[x], y

    def getDenseFeatures(self, training_set):
        num_instances = len(training_set)

        x = np.zeros((num_instances, len(self.featureDict)))
        y = np.zeros((num_instances, 1))

        for i in range(num_instances):
            instance = training_set[i]
            y[i, 0] = self.label.get(instance.label) 

            for f in instance.features().split():
                if f in self.featureDict:
                    x[i, self.featureDict.get(f)] = 1
                else:
                    continue
        return x, y

    def setUp(self, training_set, test_set):
        # Get dense features representation of dataset
        # Get training and test set
        self.x_matrix, self.y_matrix = self.getDenseFeatures(training_set)
        self.x_test, self.y_test = self.getDenseFeatures(test_set)


    def train(self, training_set, test_set):
        """Training function """
        # Set up the dataset
        self.setUp(training_set, test_set)

        num_instances, num_features = self.x_matrix.shape
        print("Start to compute gram matrix")
        K = self.gram_matrix(num_instances)
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b
        # calculate larange multipliers
        print("Start to compute largrange multipliers")
        self.lagrange_multipliers = self.compute_multipliers(K, num_instances, num_features)

        support_vector_indices = \
            self.lagrange_multipliers > 1e-5
        index = np.arange(len(self.lagrange_multipliers))[support_vector_indices]
        self.lagrange_multipliers = self.lagrange_multipliers[support_vector_indices]
        self.support_vectors = self.x_matrix[support_vector_indices]
        self.support_vector_labels = self.y_matrix[support_vector_indices]

        print("Start to compute b")
        # compute b
        self.b = 0
        for n in range(len(self.lagrange_multipliers)):
            self.b += self.support_vector_labels[n]
            self.b -= np.sum(self.lagrange_multipliers * self.support_vector_labels * K[index[n],support_vector_indices])
        self.b /= len(self.lagrange_multipliers)


        # linear kernel weight vector
        if self.kernel == Kernel.linear:
            print("Start to compute weight vector")
            self.w = np.zeros(num_features)
            for n in range(len(self.lagrange_multipliers)):
                self.w += self.lagrange_multipliers[n] * self.support_vector_labels[n] * self.support_vectors[n]
        else:
            self.w = None

        print("test accuracy")
        self.accuracy()

    def gram_matrix(self, num_instances):
        K = np.zeros((num_instances, num_instances))
        for i, x_i in enumerate(self.x_matrix):
            for j, x_j in enumerate(self.x_matrix):
                K[i, j] = self.kernel(x_i, x_j)
        return K

    def compute_multipliers(self, K, num_instances, num_features):
        """compute the lagrange_multipliers reference from https://github.com/ajtulloch/svmpy """

        P = cvxopt.matrix(np.outer(self.y_matrix, self.y_matrix) * K)
        q = cvxopt.matrix(-1 * np.ones(num_instances))
        A = cvxopt.matrix(self.y_matrix, (1, num_instances))
        b = cvxopt.matrix(0.0)

        # -a_i \leq 0
        # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
        G_std = cvxopt.matrix(np.diag(np.ones(num_instances) * -1))
        h_std = cvxopt.matrix(np.zeros(num_instances))

        # # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(num_instances)))
        h_slack = cvxopt.matrix(np.ones(num_instances) * self.c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        return np.ravel(solution['x'])


    def classify(self):
        """
        Computes the SVM prediction on the given features x.
        """
        if self.w is not None:
            return np.sign(np.dot(self.x_test, self.w) + self.b)
        else:
            y_predict = np.zeros(len(self.x_test))
            for i in range(len(self.x_test)):
                temp = 0
                for lagrange, vector_label, vector in zip(self.lagrange_multipliers, self.support_vector_labels, self.support_vectors):
                    temp += lagrange * vector_label * self.kernel(self.x_test[i], vector)
                y_predict[i] = temp
            return y_predict + self.b

    def accuracy(self):
        """Test the accuracy of the test set """
        correct = float(np.sum(self.classify() == self.y_test.T))
        print(correct / len(self.y_test))



