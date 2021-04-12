# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:36:12 2020

@author: MatteoRisso
"""

import os
import numpy as np
import pickle

def data_generator(variant):
    if os.path.exists(variant + "/corpus"):
        corpus = pickle.load(open(variant + '/corpus', 'rb'))
    else:
        corpus = Corpus(variant)
        pickle.dump(corpus, open(variant + '/corpus', 'wb'))
    return corpus


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.ndarray(shape=(tokens))
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def batchify(data, batch_size):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.shape[0] // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:nbatch * batch_size]
    # Evenly divide the data across the batch_size batches.
    data = np.reshape(data, (batch_size, -1))
    
    return data


def get_batch(source, i, args_seq_len, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args_seq_len, source.shape[1] - 1 - i)
    data = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len]
    return data, target