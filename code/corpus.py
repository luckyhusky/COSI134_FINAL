"""
For the purpose of classification, a corpus is defined as a collection of
label and documents. To the classifier they are merely instances with features
"""
from abc import ABCMeta, abstractmethod
from csv import reader as csv_reader
from glob import glob
import enchant

class Document(object):
    """A document completely characterized by its features """
    max_display_data = 10

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
        self.datafiles = glob(datafiles)
        for datafile in self.datafiles:
            self.load(datafile, document_class)

    # Act as a mutable container for documents.
    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    # def __getitem__(self, key):
    #     return self.documents[key]
    #
    # def __setitem__(self, key, value):
    #     self.documents[key] = value
    #
    # def __delitem__(self, key):
    #     del self.documents[key]

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
        with open(datafile, "r") as file:
            # Get all the documents in English"""
            dictionary = enchant.Dict("en_US")
            for publishID, videoID, userID, comment, label in unicode_csv_reader(file):
                # if dictionary.check(comment):
                #     data = [publishID, videoID, userID, comment]
                #     self.documents.append(document_class(data, label, datafile))
                data = [publishID, videoID, userID, comment]
                print(data)
                self.documents.append(document_class(data, label, datafile))

class CommentsCorpus(CSVCorpus):
    """A collection of blog posts, labeled by author gender. See the paper
    "Improving Gender Classification of Blog Authors" by Mukherjee and Liu
    <http://www.cs.uic.edu/~liub/publications/EMNLP-2010-blog-gender.pdf>
    for details and some impressive results."""

    def __init__(self, datafiles="../data/youtube_comments_20120117.csv",
                 document_class=Document):
        super(CommentsCorpus, self).__init__(datafiles, document_class)
