# Logistic Regression Binary Classifier
# I use bog-of-word as my baisc features and several other features
# change the features as you need
'''
a. Document statistics: the log of the total number of words in the document
b. Binary word features: 1 for if the punctuation “!” is in the document, 0 otherwise.
c. Binary word features: 1 for if the punctuation “?” is in the document, 0 otherwise.
I collect the top 100 most frequently use word for each class.
d. Binary word features: 1 for if top 100 most frequently use word for positive class is in the document, 0 otherwise.
e. Binary word features: 1 for if top 100 most frequently use word for negtive class is in the document, 0 otherwise.
features b, c, d, e are useless, just for example
'''

import os
from typing import Sequence, DefaultDict, Dict

import numpy as np
from collections import defaultdict
from math import ceil, log
from random import Random
import string
from scipy.special import expit # logistic (sigmoid) function
from collections import Counter
from nltk.corpus import stopwords


class LogisticRegression():

    def __init__(self):
        self.class_dict = {}
        self.feature_dict = {}
        self.n_features = None
        self.theta = None # weights (and bias)
        self.senti = {}


    def make_dicts(self, train_set_path: str) -> None:
        '''
        Given a training set, fills in self.class_dict and self.feature_dict
        Also sets the number of features self.n_features and initializes the
        parameter vector self.theta.
        '''
        # iterate over training documents
        n_label = 0
        vocab = 0
        sentidic = defaultdict(Counter)
        stop_words = set(stopwords.words('english'))
        for root, dirs, files in os.walk(train_set_path):
            for name in files:

                # this if statement is necessary for MacOs
                if name == ".DS_Store":
                    continue

                with open(os.path.join(root, name), encoding="utf8", errors="ignore") as f:
                    label = os.path.basename(root)

                    #suggest positive for 1 and negtive for 0
                    if label not in self.class_dict:
                        self.class_dict[label] = n_label
                        n_label += 1

                    #create word count features (bag-of-word)
                    content = f.read().split()
                    for word in content:
                        if word not in string.punctuation:
                            if word not in self.feature_dict:
                                self.feature_dict[word] = vocab
                                vocab += 1

                            # prepare for top 100 most frequently use word without stop word features
                            if word not in stop_words:
                                sentidic[label][word] += 1
        for l in sentidic:
            self.senti[l] = [word for word,cnt in sentidic[l].most_common(100)]

        self.n_features = len(self.feature_dict) + 5
        self.theta = np.zeros(self.n_features + 1)


    def load_data(self, data_set_path: str):
        '''
        Loads a dataset. Returns a list of filenames, and dictionaries
        of classes and documents such that:
        classes[filename] = class of the document
        documents[filename] = feature vector for the document (use self.featurize)
        '''
        filenames = []
        classes = dict()
        documents = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set_path):
            for name in files:

                # this if statement is necessary for MacOs
                if name == ".DS_Store":
                    continue

                with open(os.path.join(root, name), encoding="utf8", errors="ignore") as f:
                    filenames.append(name)
                    label = os.path.basename(root)
                    classes[name] = self.class_dict[label]
                    content = f.read().split()
                    docs = []
                    for word in content:
                        if word not in string.punctuation:
                            docs.append(word)

                        #prepare for extra punctuation features
                        elif word == "!" or "?":
                            docs.append(word)

                    documents[name] = self.featurize(docs)
        return filenames, classes, documents


    def featurize(self, document: Sequence[str]) -> np.array:
        '''
        Given a document (as a list of words), returns a feature vector.
        '''
        vector = np.zeros(self.n_features + 1)   # + 1 for bias
        #for word counts features
        word_counts = defaultdict(int)
        for word in document:
            word_counts[word] += 1

            #for extra punctuation features
            if word == "!":
                vector[-3] = 1
            elif word == "?":
                vector[-4] = 1
            #for top 100 most frequently use word without stop word features
            n = -6
            for s in self.senti:
                if word in self.senti[s]:
                    vector[n] = 1
                    continue
                n += 1

        for word, count in word_counts.items():
            if word in self.feature_dict:
                vector[self.feature_dict[word]] = count
        # for log(word count of doc)
        vector[-2] = log(len(document))
        vector[-1] = 1   # bias
        return vector


    def train(self, train_set_path: str, batch_size=0, n_epochs=0, eta=0.01) -> None:
        '''
        Trains a logistic regression classifier on a training set.
        '''
        filenames, classes, documents = self.load_data(train_set_path)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
                size = len(minibatch)
                # create and fill in matrix x and vector y
                x = np.zeros((size, self.n_features + 1))
                y = np.zeros(size)
                for k in range(size):
                    file = minibatch[k]
                    y[k] = classes[file]
                    for j in range(len(documents[file])):
                        x[k, j] = documents[file][j]
                # compute y_hat
                y_hat = expit(np.dot(x, self.theta))
                # update loss
                loss += -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
                # compute gradient
                gradient = (1 / size) * np.dot(np.transpose(x), (y_hat - y))
                # update weights (and bias)
                self.theta -= eta * gradient
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)


    def test(self, dev_set_path: str) -> DefaultDict[str, Dict[str, int]]:
        '''
        Tests the classifier on a development or test set.
        Returns a dictionary of filenames mapped to their correct and predicted classes
        '''
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set_path)
        for name in filenames:
            # get most likely class (recall that P(y=1|x) = y_hat)
            y_hat = expit(np.dot(documents[name], self.theta))
            results[name]['correct'] = classes[name]
            if y_hat > 0.5:
                results[name]['predicted'] = 1
            else:
                results[name]['predicted'] = 0
        return results


    def evaluate(self, results: DefaultDict[str, Dict[str, int]]) -> None:
        '''
        Given results, calculates the following:
        Precision, Recall, F1 for each class
        Accuracy overall
        Also, prints evaluation metrics in readable format.
        '''
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))
        for filename, result_dict in results.items():
            cindex = result_dict['correct']
            pindex = result_dict['predicted']
            confusion_matrix[pindex, cindex] += 1
        row_sum = np.sum(confusion_matrix, axis=0)
        col_sum = np.sum(confusion_matrix, axis=1)
        cc = 0
        for label in self.class_dict:
            label_index = self.class_dict[label]
            tp = confusion_matrix[label_index, label_index]
            cc += tp
            tfp = row_sum[label_index]
            tfc = col_sum[label_index]
            precision = 0 if tfp == 0 else tp / tfp
            recall = 0 if tfc == 0 else tp / tfc
            f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            print(f'{label}:')
            print(f'    precision: {precision}')
            print(f'    recall: {recall}')
            print(f'    f1: {f1}')
        accuracy = cc / np.sum(row_sum)
        print(f'Overall Accuracy: {accuracy}')
        pass

if __name__ == '__main__':
    lr = LogisticRegression()
    # make sure these point to the right directories
    lr.make_dicts('path/train')
    #change your hyperparameters are you need
    lr.train('path/train', batch_size=10, n_epochs=10, eta=0.005)
    results = lr.test('path/dev')
    lr.evaluate(results)
