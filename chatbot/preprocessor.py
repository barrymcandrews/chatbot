from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from typing import Set, List
import json
from keras_preprocessing.text import tokenizer_from_json
from os.path import exists
import os
import copy
import re
from enum import IntEnum


class Token(IntEnum):
    STX = 1
    ETX = 2
    UNK = 3

class TextPreprocessor():
    def __init__(self, dictionary, max_num_tokens):
        self.dictionary = dictionary
        self.max_num_tokens = max_num_tokens

    def tokenize(self, words):
        return [self.dictionary.word_to_index.get(w) or Token.UNK for w in words]

    def prepare(self, words: List[str], response=False, add_start=True, add_end=True, max_len=None):
        tokenized = self.tokenize(words)
        if add_start:
            tokenized.insert(0, Token.STX)
        if add_end:
            tokenized.append(Token.ETX)

        max_len = self.max_num_tokens if max_len is None else max_len
        padding = 'post' if response else 'pre'
        return pad_sequences(sequences=[tokenized], maxlen=max_len, padding=padding, truncating=padding)[0]

