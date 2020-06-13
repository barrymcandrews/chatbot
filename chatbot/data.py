
from functools import reduce
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from convokit import Corpus, download, TextParser, TextProcessor
from itertools import tee
from tqdm import tqdm
from nltk import FreqDist
import numpy as np
import tensorflow as tf
import wget
import zipfile
import yaml
import os
import re
import json

from chatbot.preprocessor import TextPreprocessor, Token
from chatbot.dictionary import Dictionary
from chatbot.util import clean
from chatbot.sources.imessage import get_imessage_corpus

@dataclass
class ChatbotDataset():
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def save(self, filename='build/.dataset.npy'):
        print('Saving dataset...')
        np.save(filename, np.array([self.x, self.y, self.z]))
        print('Dataset saved to ' + filename)

    @staticmethod
    def load(filename='build/.dataset.npy'):
        print('Loading dataset from ' + filename)
        d = np.load(filename)
        return ChatbotDataset(d[0], d[1], d[2])


ChatbotData = Tuple[ChatbotDataset, Dictionary, int]


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def load_dataset() -> ChatbotData:
    # corpus = Corpus(filename=download("movie-corpus"))
    corpus = get_imessage_corpus()

    corpus = TextProcessor(clean, 'words').transform(corpus)

    conversations = []
    utts = []
    for path in ['data/manual/greetings.yml', 'data/manual/identity.yml', 'data/manual/meta.yml']:
        with open(path) as f:
            cs = yaml.load(f.read())['conversations']
            for c in cs:
                conversations.append([
                    clean(c[0]),
                    clean(c[1])
                ])
                utts.append(clean(c[0]))
                utts.append(clean(c[1]))

    # Dictionary
    dictionary = None
    if os.path.exists('build/dictionary.json'):
        dictionary = Dictionary.load()
    else:
        print('Building dictionary.')
        all_words = [word for u in corpus.iter_utterances() for word in u.meta['words']]
        all_words.extend([word for u in utts for word in u])
        dictionary = Dictionary.from_word_list(all_words)
        dictionary.save()

    MAX_LEN = 22
    preprocessor = TextPreprocessor(dictionary, MAX_LEN)

    # Dataset
    dataset = None
    if os.path.exists('build/.dataset.npy'):
        dataset = ChatbotDataset.load()
    else:
        corpus = TextProcessor(lambda text: preprocessor.tokenize(clean(text)), 'tokens').transform(corpus)
        print("Tokenizing data.")
        contexts = []
        responses = []
        for conversation in corpus.iter_conversations():
            all_utterances = [u for u in conversation.iter_utterances()]

            for (u1, u2) in _pairwise(all_utterances):
                if len(u1.meta['words']) < MAX_LEN \
                    and len(u2.meta['words']) < MAX_LEN \
                    and Token.UNK not in u2.meta['tokens'] \
                    and '*' not in u1.meta['words'] \
                    and u1.meta['is_from_me'] == '0' \
                    and u2.meta['is_from_me'] == '1':

                    contexts.append(u1.meta['words'])
                    responses.append(u2.meta['words'])

        for _ in range(100):
            for conversation in conversations:
                contexts.append(conversation[0])
                responses.append(conversation[1])

        for i in range(len(responses)):
            if responses[i] == Token.UNK:
                raise RuntimeError("Unknown token is in response data!")

        print("Tokenizing contexts (x)")
        x = [preprocessor.prepare(tokens) for tokens in tqdm(contexts)]
        print("Tokenizing responses (y)")
        y = [preprocessor.prepare(tokens, response=True) for tokens in tqdm(responses)]
        print("Tokenizing responses (z)")
        z = [preprocessor.prepare(tokens, response=True, add_start=False) for tokens in tqdm(responses)]

        dataset = ChatbotDataset(np.array(x), np.array(y), np.array(z))
        dataset.save()

    return (dataset, dictionary, MAX_LEN)
