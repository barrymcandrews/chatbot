from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from typing import Set
import json
from keras_preprocessing.text import tokenizer_from_json
from os.path import exists
import os
import copy
import re

STX = '<stx>'
ETX = '<etx>'

def _flat_map(list):
    return [x for y in list for x in y]


def _split(string):
        string = re.sub(r'(\.)', ' . ', string)
        string = re.sub(r'(\?)', ' ? ', string)
        string =  re.sub(r'(\!)', ' ! ' , string)
        string =  re.sub(r'[\-]{2,}', ' -- ' , string)
        string =  re.sub(r'[ \t]{2,}', ' ', string)
        return string.strip().split(' ')

class TextPreprocessor():
    def __init__(self, tokenizer, max_context_length):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length

    def get_vocabulary_size(self):
        return len(self.tokenizer.word_index) + 1

    def prepare(self, string: str, response=False, add_start=False, add_end=False, max_len=None):
        max_len = self.max_context_length if max_len is None else max_len
        seqs = _flat_map(self.tokenizer.texts_to_sequences(_split(string)))
        if add_start:
            for words in seqs:
                words.insert(0, STX)
        if add_end:
            for words in seqs:
                words.append(ETX)
        padding = 'post' if response else 'pre'
        return pad_sequences(sequences=[seqs], maxlen=max_len, padding=padding, truncating=padding)

    def save(self, folderName='build/'):
        os.makedirs(folderName, exist_ok=True)
        with open(folderName + 'tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(self.tokenizer.to_json())

        with open(folderName + 'build.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'maxContextLength': self.max_context_length
            }))

    @staticmethod
    def load(folderName='build/'):
        tokenizer = None
        max_context_length = None
        with open(folderName + 'tokenizer.json') as f:
            tokenizer = tokenizer_from_json(f.read())

        with open(folderName + 'build.json') as f:
            max_context_length = json.load(f)['maxContextLength']

        return TextPreprocessor(tokenizer, max_context_length)

    @staticmethod
    def build_exists(folderName='build/'):
        return exists(folderName + 'build.json') and exists(folderName + 'tokenizer.json')


class PreprocessorBuilder():
    def __init__(self):
        self.vocab = set([STX, ETX])
        self.max_context_length = 0

    def fit(self, string):
        words = _split(string)
        self.vocab = self.vocab.union(words)
        self.max_context_length = max(self.max_context_length, len(words))

    def build(self) -> TextPreprocessor:
        tokenizer = Tokenizer(filters=[])
        tokenizer.fit_on_texts(self.vocab)
        return TextPreprocessor(tokenizer, self.max_context_length)
