# Part-of-speech Tagging with Structured Perceptrons
# I use Brown dataset when write this code
# tagged sentences in the following format: [<word>/<tag>]

import os
import pickle

import numpy as np
from collections import defaultdict
from random import Random

class POSTagger():

    def __init__(self):
        self.tag_dict = {}
        self.word_dict = {}
        self.initial = None
        self.transition = None
        self.emission = None
        self.unk_index = np.inf


    def make_dicts(self, train_set):
        '''
        Fills in self.tag_dict and self.word_dict, based on the training data.
        '''
        # Iterate over training documents
        tag_index = 0
        word_index = 0
        for root, dirs, files in os.walk(train_set):
            for name in files:

                # this if statement is necessary for MacOs
                if name == ".DS_Store":
                    continue

                with open(os.path.join(root, name)) as f:
                    content = f.read().split()
                    for word in content:
                        long = len(word)
                        split = 0
                        for i in range(long):
                            if word[i] == "/":
                                split = i
                        x = word[0:split]
                        tag = word[split+1:]
                        if tag not in self.tag_dict:
                            self.tag_dict[tag] = tag_index
                            tag_index += 1
                        if x not in self.word_dict:
                            self.word_dict[x] = word_index
                            word_index += 1




    def load_data(self, data_set):
        '''
        Loads a dataset, returns a list of sentence_ids, and
        dictionaries of tag_lists and word_lists.
        '''
        sentence_ids = []
        sidx = 0
        tag_lists = dict()
        word_lists = dict()
        # Iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                if name == ".DS_Store":
                    continue
                with open(os.path.join(root, name), 'r') as f:
                    # Split documents into sentences here
                    sentences = [line.strip() for line in f if line.strip()]
                    for sentence in sentences:
                        tlist = []
                        wlist = []
                        words = sentence.split()
                        for word in words:
                            long = len(word)
                            split = 0
                            for i in range(long):
                                if word[i] == "/":
                                    split = i
                            x = word[0:split]
                            tag = word[split + 1:]
                            if tag in self.tag_dict:
                                tlist.append(self.tag_dict[tag])
                            else:
                                tlist.append(self.unk_index)
                            if x in self.word_dict:
                                wlist.append(self.word_dict[x])
                            else:
                                wlist.append(self.unk_index)
                        tag_lists[sidx] = tlist
                        word_lists[sidx] = wlist
                        sentence_ids.append(sidx)
                        sidx += 1
        return sentence_ids, tag_lists, word_lists


    def viterbi(self, sentence):
        '''
        Implements the Viterbi algorithm.
        Use v and backpointer to find the best_path.
        '''
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        # Initialization step
        if sentence[0] != self.unk_index:
            v[:, 0] = self.initial[:] + self.emission[sentence[0], :]
        else:
            v[:, 0] = self.initial[:]
        backpointer[:, 0] = -1
        # Recursion step
        for t in range(1, T):
            word = sentence[t]
            if word != self.unk_index:
                temp = self.emission[word, :] + self.transition[:, :] + np.reshape(v[:, t-1], (N, 1))
            else:
                temp = 0 + self.transition[:, :] + np.reshape(v[:, t-1], (N, 1))
            best_tag = np.argmax(temp, axis=0)
            backpointer[:, t] = best_tag
            max_values = np.take_along_axis(temp, best_tag[np.newaxis, :], axis=0)
            v[:, t] = max_values
        # Termination step
        best_end = np.argmax(v[:, T-1])
        pointer = best_end
        best_path = [best_end]
        for t in range(T-1, 0, -1):
            prev = backpointer[pointer, t]
            best_path.append(prev)
            pointer = prev
        best_path = best_path[::-1]
        return best_path


    def train(self, train_set, learning_rate=1):
        '''
        Trains a structured perceptron part-of-speech tagger on a training set.
        '''
        self.make_dicts(train_set)
        sentence_ids, tag_lists, word_lists = self.load_data(train_set)
        Random(0).shuffle(sentence_ids)
        self.initial = np.zeros(len(self.tag_dict))
        self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
        self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        for i, sentence_id in enumerate(sentence_ids):
            y_hat = self.viterbi(word_lists[sentence_id])
            y = tag_lists[sentence_id]
            w = word_lists[sentence_id]
            if y_hat != y:
                for k in range(len(y_hat)):
                    self.emission[w[k], y_hat[k]] -= 1 * learning_rate
                    self.emission[w[k], y[k]] += 1 * learning_rate
                    if k == 0:
                        self.initial[y_hat[k]] -= 1 * learning_rate
                        self.initial[y[k]] += 1 * learning_rate
                    else:
                        self.transition[y_hat[k-1], y_hat[k]] -= 1 * learning_rate
                        self.transition[y[k - 1], y[k]] += 1 * learning_rate
            # Prints progress of training
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')


    def test(self, dev_set):
        '''
        Tests the tagger on a development or test set.
        '''
        results = defaultdict(dict)
        sentence_ids, tag_lists, word_lists = self.load_data(dev_set)
        for i, sentence_id in enumerate(sentence_ids):
            results[sentence_id]['correct'] = tag_lists[sentence_id]
            results[sentence_id]['predicted'] = self.viterbi(word_lists[sentence_id])
            # your code here
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return results


    def evaluate(self, results):
        '''
        Given results, calculates overall accuracy.
        '''
        correct_count = 0
        total_count = 0
        for key, value in results.items():
            for tag_correct, tag_predicted in zip(value['correct'], value['predicted']):
                total_count += 1
                if tag_correct == tag_predicted:
                    correct_count += 1
        accuracy = correct_count / total_count
        return accuracy


if __name__ == '__main__':
    pos = POSTagger()
    # Make sure train and test point to the right directories
    # Change the learning rate as you need
    pos.train('path/train', learning_rate=1)
    # Writes the POS tagger to a file
    #with open('pos_tagger.pkl', 'wb') as f:
    #    pickle.dump(pos, f)
    #Reads the POS tagger from a file
    #with open('pos_tagger.pkl', 'rb') as f:
    #    pos = pickle.load(f)
    results = pos.test('path/dev')
    print('Accuracy:', pos.evaluate(results))