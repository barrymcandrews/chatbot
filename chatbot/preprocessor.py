from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from typing import Set
import json
from keras_preprocessing.text import tokenizer_from_json
import os

def flat_map(list):
    return [x for y in list for x in y]


class TextPreprocessor():
    def __init__(self, vocab: Set[str], max_context_length: int):
        self.tokenizer = Tokenizer(filters=[])
        self.tokenizer.fit_on_texts(vocab)
        self.max_context_length = max_context_length
        print("Max context lenght: " + str(max_context_length))
        print("Vocabulary size: " + str(len(vocab)))

    def prepare_texts(self, texts):
        return pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.max_context_length)

    def prepare(self, string: str):
        print(string)
        seqs = flat_map(self.tokenizer.texts_to_sequences(string.split(' ')))
        print(seqs)
        return pad_sequences(sequences=[seqs], maxlen=self.max_context_length)

    def save(self, folderName='build/'):
        os.makedirs(folderName)
        with open(folderName + 'tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(self.tokenizer.to_json())

        with open(folderName + 'build.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'maxContextLength': self.max_context_length
            }))

    @staticmethod
    def load(folderName='build/'):
        recovered = TextPreprocessor(set(), 0)
        with open(folderName + 'tokenizer.json') as f:
            recovered.tokenizer = tokenizer_from_json(f.read())

        with open(folderName + 'build.json') as f:
            recovered.max_context_length = json.load(f)['maxContextLength']

        return recovered





