import os
import re
from functools import reduce
from dataclasses import dataclass
from typing import List, Set, Tuple
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from chatbot.preprocessor import TextPreprocessor, PreprocessorBuilder
from convokit import Corpus, download
from convokit.text_processing.textProcessor import TextProcessor
from convokit.text_processing.textParser import TextParser
from itertools import tee
import json
from tqdm import tqdm


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


ChatbotData = Tuple[ChatbotDataset, TextPreprocessor]


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def load_movie_dataset() -> ChatbotData:
    corpus = Corpus(filename=download("movie-corpus"))

    # Preprocessor
    preprocessor = None
    if TextPreprocessor.build_exists():
        print("Build found. Using vocabulary from previous build.")
        preprocessor = TextPreprocessor.load()
    else:
        print("Building preprocessor for vocabulary.")
        builder = PreprocessorBuilder()
        all_utterances = [u for u in corpus.iter_utterances()]
        for utterance in tqdm(all_utterances):
            builder.fit(utterance.text)
        preprocessor = builder.build()
        preprocessor.save()
        print("Preprocessor build saved.")

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
                contexts.append(u1.text)
                responses.append(u2.text)

        print("Tokenizing contexts (x)")
        x = [preprocessor.prepare(text) for text in tqdm(contexts)]
        print("Tokenizing responses (y)")
        y = [preprocessor.prepare(text, response=True, add_start=True) for text in tqdm(responses)]
        print("Tokenizing responses (z)")
        z = [preprocessor.prepare(text, response=True, add_end=True) for text in tqdm(responses)]

        dataset = ChatbotDataset(np.array(x), np.array(y), np.array(z))
        dataset.save()

    return (dataset, preprocessor)
