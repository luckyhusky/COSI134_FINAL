"""
For the purpose of classification, a corpus is defined as a collection of
label and documents. To the classifier they are merely instances with features
"""
from abc import ABCMeta, abstractmethod
from csv import reader as csv_reader
from glob import glob
from nltk.tokenize import word_tokenize
# enchant is a dictionary to check english words
import enchant
# create a new dataset with all english comments
import xlwt
import csv
# check if the prepared data file exist
import os.path
import numpy as np

class Document(object):
    """A document completely characterized by its features """
    max_display_data = 10

    # @param data  data = [publishID, videoID, userID, comment]
    # @param label label = [true, false]
    def __init__(self, data, label = None, source = None):
        self.data = data
        self.label = label
        self.source = source

    def __repr__(self):
        return ("<%s: %s>" % (self.label, self.abbrev()) if self.label else
                "%s" % self.abbrev())

    def abbrev(self):
        return (self.data if len(self.data) < self.max_display_data else
                self.data[0:self.max_display_data] + "...")

    def features(self):
        """A list of features that characterized this document."""
        return [self.data]


class Corpus(object):
    """An abstract collection of documents."""
    __metaclass__ = ABCMeta

    def __init__(self, datafiles, document_class = Document):
        self.documents = []
        self.label = {} # Store all the label in the dataset
        self.featureDict = {} # store all the feature in the featureDict
        # self.features = [] # instances with sparse or dense feaure representation
        # self.labels = [] #labels of all the instances
        self.datafiles = glob(datafiles)
        for datafile in self.datafiles:
            self.load(datafile, document_class)

    # Act as a mutable container for documents.
    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def __getitem__(self, key):
        return self.documents[key]

    def __setitem__(self, key, value):
        self.documents[key] = value

    def __delitem__(self, key):
        del self.documents[key]

    @abstractmethod
    def load(self, datafile, document_class):
        """Make labeled document instances for the data in a file."""
        pass

class CSVCorpus(Corpus):
    """A corpus encodes as a comma-separated-value file """
    def load(self, datafile, document_class, encoding = "utf-8"):
        """Make a document from each row of a CSV datafile
        Assumes data, label ordering and UTF-8 encoding """
        def unicode_csv_reader(csvfile, *args, **kwargs):
            for row in csv_reader(csvfile, *args, **kwargs):
                yield[unicode(cell, encoding) for cell in row]
                        

        # if the prepared dataset exists, read the prepared one
        if(os.path.isfile("../data/prepared60000.csv")):
             with open("../data/prepared60000.csv", "r") as file:
                for comment in unicode_csv_reader(file):
                    self.documents.append(document_class(comment[0], comment[1], datafile))
                print(len(self.documents))
                self.featurize()
        else:
            with open(datafile, "r") as file:
                # Get all the documents in English"""
                dictionary = enchant.Dict("en_US")
                for publishID, videoID, userID, comment, label in unicode_csv_reader(file):
                    # Check if the sentence is english sentence
                    # Tokenize the word first
                    comment = self.words_tokenize(comment)
                    commentlen = len(comment)                
                    engWord = 0.0
                    for i in range(commentlen):
                        if dictionary.check(comment[i]):
                            engWord += 1
                    # If the percentage of the english word is greater than 0.6 than put it in the document
                    if((engWord / commentlen) > 0.6):
                        self.documents.append(document_class(comment, label, datafile))
            file.close()

            # write into a new csv file for training             
            with open("../data/prepared.csv", "wb") as csvfile:
                writer = csv.writer(csvfile)
                for instance in self.documents:
                    writer.writerow([instance.data, instance.label])
            csvfile.close()
            print("The new dataset is prepared, please run again")

       

    def words_tokenize(self, text, language='english'):
        """Trivially tokenize """
        return word_tokenize(text)

    def featurize(self):
        """Create featureDict to help with create sparse features for each instance"""
        labelInx = -1
        featureInx = 0

        # create feature dictionary for all the instances
        for instance in self.documents:
            if instance.label not in self.label:
                # self.featureDict[instance.label] = {}
                self.label[instance.label] = labelInx
                labelInx += 2

            # get all the feature from one certain distance and create sparse representation of the instance  
            for f in instance.features().split():
                if f not in self.featureDict:
                    self.featureDict[f] = featureInx
                    featureInx += 1

class CommentsCorpus(CSVCorpus):
    """A collection of comments post"""
    def __init__(self, datafiles="../data/comments300000.csv",
                 document_class=Document):
        super(CommentsCorpus, self).__init__(datafiles, document_class)
