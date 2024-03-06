# Naive Bayes in Numpy

import os
import numpy as np
from collections import defaultdict
import string
from math import log

def safe_log(n) -> float:
    return float("-inf") if n == 0.0 else log(n)

class NaiveBayes():

    def __init__(self):
        self.class_dict = {}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    '''
    def train(self, train_set):
        # iterate over training documents
        word_count = {}
        n_label = 0
        vocab = 0
        Ndoc = {}
        for root, dirs, files in os.walk(train_set):
            for name in files:

                # this if statement is necessary for MacOs
                if name == ".DS_Store":
                    continue

                with open(os.path.join(root, name), encoding="utf8", errors="ignore") as f:
                    # collect class counts and feature counts
                    label = os.path.basename(root)
                    if label not in self.class_dict:
                        self.class_dict[label] = n_label
                        Ndoc[label] = 1
                        n_label += 1
                        word_count[label] = {}
                    else:
                        Ndoc[label] += 1
                    content = f.read().split()
                    for word in content:
                        if word not in string.punctuation:
                            if word in word_count[label]:
                                word_count[label][word] += 1
                            else:
                                word_count[label][word] = 1
                                if word not in self.feature_dict:
                                    self.feature_dict[word] = vocab
                                    vocab += 1
        self.prior = np.zeros(len(self.class_dict))
        for label in self.class_dict:
            index = self.class_dict[label]
            self.prior[index] = safe_log(Ndoc[label]/sum(Ndoc.values()))
        self.likelihood = np.zeros((vocab, len(self.class_dict)))
        for label in self.class_dict:
            label_words = sum(word_count[label].values())
            for word in self.feature_dict:
                wc = word_count[label][word] if word in word_count[label] else 0
                self.likelihood[self.feature_dict[word], self.class_dict[label]] = safe_log((wc + 1)/(label_words + vocab))


    '''
    Tests the classifier on a development or test set.
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:

                # this if statement is necessary for MacOs
                if name == ".DS_Store":
                    continue

                with open(os.path.join(root, name)) as f:
                    # create feature vectors for each document
                    label = os.path.basename(root)
                    content = f.read().split()
                    vector = np.zeros(len(self.feature_dict))
                    for word in content:
                        if word not in string.punctuation:
                            if word in self.feature_dict:
                                index = self.feature_dict[word]
                                vector[index] += 1
                    dot_product = np.dot(vector, self.likelihood)
                    argmax = dot_product + self.prior
                    k = list(self.class_dict.keys())
                    perdict = k[np.argmax(argmax, axis=None)]
                    results[name] = {'correct': label, 'perdicted': perdict}
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        for filename, result_dict in results.items():
            cindex = self.class_dict[result_dict['correct']]
            pindex = self.class_dict[result_dict['perdicted']]
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

if __name__ == '__main__':
    nb = NaiveBayes()
    #make sure these point to the right directories
    nb.train('path/train')
    results = nb.test('path/dev')
    nb.evaluate(results)
