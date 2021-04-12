# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:50:35 2020

@author: MatteoRisso
"""

import os
import numpy as np
import pickle
import observations
import unidecode
from collections import Counter


def data_generator(variant):
    file, testfile, valfile = getattr(observations, 'ptb')('data/')
    file_len = len(file)
    valfile_len = len(valfile)
    testfile_len = len(testfile)
    corpus = Corpus(file + " " + valfile + " " + testfile)

    #############################################################
    # Use the following if you want to pickle the loaded data
    #
    # pickle_name = "{0}.corpus".format(args.dataset)
    # if os.path.exists(pickle_name):
    #     corpus = pickle.load(open(pickle_name, 'rb'))
    # else:
    #     corpus = Corpus(file + " " + valfile + " " + testfile)
    #     pickle.dump(corpus, open(pickle_name, 'wb'))
    #############################################################

    return file, file_len, valfile, valfile_len, testfile, testfile_len, corpus


def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()

    def add_word(self, char):
        self.counter[char] += 1

    def prep_dict(self):
        for char in self.counter:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self, string):
        self.dict = Dictionary()
        for c in string:
            self.dict.add_word(c)
        self.dict.prep_dict()


def char_tensor(corpus, string):
    tensor = np.zeros(len(string))
    for i in range(len(string)):
        tensor[i] = corpus.dict.char2idx[string[i]]
    return tensor


def batchify(data, batch_size):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.shape[0] // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:nbatch * batch_size]
    # Evenly divide the data across the batch_size batches.
    data = np.reshape(data, (batch_size, -1))
    
    return data


def get_batch(source, start_index, args_seq_len):
    seq_len = min(args_seq_len, source.shape[1] - 1 - start_index)
    end_index = start_index + seq_len
    inp = source[:, start_index:end_index]
    target = source[:, start_index+1:end_index+1]
    return inp, target

