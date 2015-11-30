from corpus import Document, CommentsCorpus
from unittest import TestCase, main


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
    def split_comments_corpus(self, document_class):
        """Split the blog post corpus into training, dev and test sets """
        comments = CommentsCorpus(document_class=document_class)
        # for c in comments:
        #     print(c.getComment())

    def test_comments_bag(self):
        """Classify comments using bag-of-words"""
        self.split_comments_corpus(BagOfWords)

if __name__ == '__main__':
    main(verbosity=2)
