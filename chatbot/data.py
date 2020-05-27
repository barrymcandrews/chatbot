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
    x: np.ndarray = np.array([], dtype=np.int32)
    y: np.ndarray = np.array([], dtype=np.int32)
    z: np.ndarray = np.array([], dtype=np.int32)

    def append(self, x, y, z):
        self.x = np.append(self.x, [x])
        self.y = np.append(self.y, [y])
        self.z = np.append(self.z, [z])


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
        all_utterances = [u for u in conversation.iter_utterances()]
        all_utterances.reverse()
        for (u1, u2) in _pairwise(all_utterances):
            dataset.append(
                x=preprocessor.prepare(u1.text),
                y=preprocessor.prepare(u2.text, response=True, add_start=True),
                z=preprocessor.prepare(u2.text, response=True, add_end=True)
            )

    return (dataset, preprocessor)
