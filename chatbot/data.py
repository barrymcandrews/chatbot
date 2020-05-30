import os
import re
from functools import reduce
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from chatbot.preprocessor import TextPreprocessor
from convokit import Corpus, download
from convokit.text_processing.textProcessor import TextProcessor
from convokit.text_processing.textParser import TextParser
from itertools import tee
import json
from tqdm import tqdm
from nltk import FreqDist
import wget
import zipfile

STX = 1
ETX = 2
UNK = 3

@dataclass
class ChatbotDataset():
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def save(self, filename='build/dataset.npy'):
        print('Saving dataset...')
        np.save(filename, np.array([self.x, self.y, self.z]))
        print('Dataset saved to ' + filename)

    @staticmethod
    def load(filename='build/dataset.npy'):
        d = np.load(filename)
        return ChatbotDataset(d[0], d[1], d[2])


@dataclass
class Dictionary():
    index_to_word: List[str]
    word_to_index: Dict[str, int]
    embeddings_matrix: np.ndarray = None

    def size(self):
        return len(self.index_to_word)

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


ChatbotData = Tuple[ChatbotDataset, Dictionary, int]


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _split(string):
        string = re.sub(r'<[^>]*>|\*|\[|\]|"', ' ', string)
        string = re.sub(r'(,)', ' , ', string)
        string = re.sub(r'(;)', ' ; ', string)
        string = re.sub(r'(\.)', ' . ', string)
        string = re.sub(r'(\?)', ' ? ', string)
        string = re.sub(r'(\!)', ' ! ' , string)
        string = re.sub(r'[\-]', ' ' , string)
        string = re.sub(r'[\-]{2,}', ' -- ' , string)
        string = re.sub(r'[ \t]{2,}', ' ', string)
        return string.strip().lower().split(' ')


def load_movie_dataset() -> ChatbotData:
    corpus = Corpus(filename=download("movie-corpus"))

    corpus = TextProcessor(_split, 'words').transform(corpus)

    # Dictionary
    dictionary = None
    if os.path.exists('build/dictionary.json'):
        dictionary = Dictionary.load()
    else:
        print('Building dictionary.')
        all_words = [word for u in corpus.iter_utterances() for word in u.meta['words']]
        word_freq = FreqDist(all_words)
        print("Found %d unique word tokens." % len(word_freq.items()))

        vocab = word_freq.most_common(7000 - 4)
        index_to_word = ['', '<stx>', '<etx>', '<unk>']
        index_to_word.extend([x[0] for x in vocab])
        word_to_index = {w: i for i, w in enumerate(index_to_word)}
        dictionary = Dictionary(index_to_word, word_to_index)
        dictionary.save()

    MAX_LEN = 50
    preprocessor = TextPreprocessor(dictionary, MAX_LEN)

    # Embeddings
    if not os.path.exists('build/glove/glove.6B.200d.txt'):
        print("Glove embeddings not found. Downloading...")
        wget.download('http://nlp.stanford.edu/data/wordvecs/glove.6B.zip', out='build')
        with zipfile.ZipFile('build/glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall('build/glove')

    print("\nLoading glove embeddings")
    embeddings_index = {}
    with open(os.path.join('build/glove/glove.6B.200d.txt')) as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors in glove file.' % len(embeddings_index))
    embeddings_matrix = np.zeros((dictionary.size(), 200))

    # Using the Glove embedding:
    print("Mapping embeddings to dictionary")
    for word, i in tqdm(dictionary.word_to_index.items()):
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embeddings_matrix[i] = embedding_vector
    dictionary.embeddings_matrix = [embeddings_matrix]

    # Dataset
    dataset = None
    if os.path.exists('build/dataset.npy'):
        dataset = ChatbotDataset.load()
    else:
        print("Tokenizing data.")
        contexts = []
        responses = []
        for conversation in corpus.iter_conversations():
            all_utterances = [u for u in conversation.iter_utterances()]
            all_utterances.reverse()

            for (u1, u2) in _pairwise(all_utterances):
                if len(u2.meta['words']) < 49 and '<unk>' not in u2.meta['words']:
                    contexts.append(u1.meta['words'])
                    responses.append(u2.meta['words'])

        print("Tokenizing contexts (x)")
        x = [preprocessor.prepare(tokens) for tokens in tqdm(contexts)]
        print("Tokenizing responses (y)")
        y = [preprocessor.prepare(tokens, response=True, add_start=True) for tokens in tqdm(responses)]
        print("Tokenizing responses (z)")
        z = [preprocessor.prepare(tokens, response=True, add_end=True) for tokens in tqdm(responses)]

        print("")

        dataset = ChatbotDataset(np.array(x), np.array(y), np.array(z))
        dataset.save()

    return (dataset, dictionary, MAX_LEN)
