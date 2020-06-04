from typing import List, Dict
from tqdm import tqdm
from nltk import FreqDist
from dataclasses import dataclass
import numpy as np
import os
import wget
import zipfile
import json


@dataclass
class Dictionary():
    index_to_word: List[str]
    word_to_index: Dict[str, int]

    def size(self):
        return len(self.index_to_word)

    def get_embeddings(self):
        if not os.path.exists('build/.glove/glove.6B.300d.txt'):
            print("Glove embeddings not found. Downloading...")
            wget.download('http://nlp.stanford.edu/data/wordvecs/glove.6B.zip', out='build')
            with zipfile.ZipFile('build/glove.6B.zip', 'r') as zip_ref:
                zip_ref.extractall('build/.glove')
            os.remove('build/glove.6B.zip')

        print("\nLoading glove embeddings")
        embeddings_index = {}
        with open(os.path.join('build/.glove/glove.6B.300d.txt')) as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors in glove file.' % len(embeddings_index))
        embeddings_matrix = np.zeros((self.size(), 300))

        # Using the Glove embedding:
        print("Mapping embeddings to dictionary")
        for word, i in tqdm(self.word_to_index.items()):
            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embeddings_matrix[i] = embedding_vector
        return [embeddings_matrix]

    def save(self, filename='build/dictionary.json'):
        print('Saving dictionary...')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'index_to_word': self.index_to_word,
                'word_to_index': self.word_to_index
            }))
        print('Dictionary saved to ' + filename)

    @staticmethod
    def load(filename='build/dictionary.json'):
       with open(filename) as f:
        data = json.loads(f.read())
        return Dictionary(data['index_to_word'], data['word_to_index'])

    @staticmethod
    def from_word_list(all_words):
        word_freq = FreqDist(all_words)
        print("Found %d unique word tokens." % len(word_freq.items()))

        vocab = word_freq.most_common(7000 - 4)
        index_to_word = ['', '<stx>', '<etx>', '<unk>']
        index_to_word.extend([x[0] for x in vocab])
        word_to_index = {w: i for i, w in enumerate(index_to_word)}
        return Dictionary(index_to_word, word_to_index)
