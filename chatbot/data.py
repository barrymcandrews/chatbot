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
from itertools import tee


@dataclass
class ChatbotDataset():
    x: np.ndarray = np.empty
    y: np.ndarray = np.empty
    z: np.ndarray = np.empty

    def append(self, x, y, z):
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)


ChatbotData = Tuple[ChatbotDataset, TextPreprocessor]


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def load_movie_dataset() -> ChatbotData:
    corpus = Corpus(filename=download("movie-corpus"))
    dataset = ChatbotDataset()

    preprocessor = None
    if TextPreprocessor.build_exists():
        print("Build found. Using vocabulary from previous build.")
        preprocessor = TextPreprocessor.load()
    else:
        print("Building preprocessor for vocabulary.")
        builder = PreprocessorBuilder()
        for utterance in corpus.iter_utterances():
            builder.fit(utterance.text)
        preprocessor = builder.build()
        preprocessor.save()
        print("Preprocessor build saved.")

    for conversation in corpus.iter_conversations():
        all_utterances = [u for u in conversation.iter_utterances()].reverse()
        if  all_utterances is None:
            continue
        for (u1, u2) in _pairwise(all_utterances):
            dataset.append(
                x=preprocessor.prepare(u1.text),
                y=preprocessor.prepare(u2.text, response=True, add_start=True),
                z=preprocessor.prepare(u2.text, response=True, add_end=True)
            )

    return (dataset, preprocessor)
