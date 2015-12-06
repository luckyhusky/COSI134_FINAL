from corpus import Document, CommentsCorpus
from unittest import TestCase, main
from random import seed, shuffle
from svm import SVM
from svm_sgd import SVM_SGD
import numpy as np
# Experiment for svm using sklearn 
from sklearn import svm
from sklearn.linear_model import SGDClassifier

class BagOfWords(Document):
    def features(self):
        return self.data

    def getPublishID(self):
        return self.data[0]

    def getVideoId(self):
        return self.data[1]

    def getUserId(self):
        return self.data[2]

    def getComment(self):
        return self.data[3]

    def getSpamLabel(self):
        return self.label


class SVMTest(TestCase):
    def set_up(self):
        corpus = CommentsCorpus(document_class=BagOfWords)
        train = corpus[0 : 4000]
        test = corpus[4000 : 4400]
        svm = SVM(corpus.label, corpus.featureDict)
        svm.setUp(train, test)
        X = svm.x_matrix
        y = svm.y_matrix
        # self.y = np.ravel(self.svm.y_matrix)
        test_X = svm.x_test
        test_y = svm.y_test
        return X, y, test_X, test_y, len(corpus.featureDict)

    def test_soft_margin_kernl_svm(self):
        pass
        # self.set_up()
        # self.svm.train(train, test)

    def test_with_skleart_sgd(self):
        """Test svm with sklearn"""
        train_x, train_y, test_x, test_y, n_features = self.set_up()
        train_y = np.ravel(train_y)
        self.svm = SGDClassifier(loss="hinge", penalty="l2")
        self.svm.fit(train_x, train_y)
        self.accuracy(test_x, test_y, "sgd")


    def test_with_sklearn_svc(self):
        train_x, train_y, test_x, test_y, n_features = self.set_up()
        train_y = np.ravel(train_y)
        self.svm = svm.SVC()
        self.svm.fit(train_x, train_y)
        self.accuracy(test_x, test_y, "svc")

    def test_with_sklearn_linearsvc(self):
        train_x, train_y, test_x, test_y, n_features = self.set_up()
        train_y = np.ravel(train_y)
        self.svm = svm.LinearSVC()
        self.svm.fit(train_x, train_y)
        self.accuracy(test_x, test_y, "linear svc")

    def test_sgd(self):
        train_x, train_y, test_x, test_y, n_features = self.set_up()
        self.svm = SVM_SGD(n_features)
        self.svm.train(train_x, train_y)
        self.svm.classify(test_x, test_y)

    def accuracy(self, test_x, test_y, classfier_type):
        result = []
        for i in test_x:
            result.append(self.svm.predict([i]))
        accuracy = float(np.sum(result == test_y)) / len(test_y)
        print("The accuracy of the" + classfier_type + "classifier is: %.3f%%" % (accuracy * 100))


if __name__ == '__main__':
    main(verbosity=2)
