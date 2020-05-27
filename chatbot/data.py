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


@dataclass
class ChatbotDataset():
    x: np.ndarray = np.array([], dtype=np.int32)
    y: np.ndarray = np.array([], dtype=np.int32)
    z: np.ndarray = np.array([], dtype=np.int32)

    def append(self, x, y, z):
        self.x = np.append(self.x, [x])
        self.y = np.append(self.y, [y])
        self.z = np.append(self.z, [z])

    def to_json(self):
        return json.dumps({
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "z": self.z.tolist(),
        })

    @staticmethod
    def from_json(json_string):
        j = json.loads(json_string)
        return ChatbotDataset(
            x=np.array(j["x"], dtype=np.int32),
            y=np.array(j["y"], dtype=np.int32),
            z=np.array(j["z"], dtype=np.int32),
        )


ChatbotData = Tuple[ChatbotDataset, TextPreprocessor]


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _reformat(string):
        string = re.sub(r'(\.)', ' . ', string)
        string = re.sub(r'(\?)', ' ? ', string)
        string =  re.sub(r'(\!)', ' ! ' , string)
        string =  re.sub(r'[\-]{2,}', ' -- ' , string)
        string =  re.sub(r'[ \t]{2,}', ' ', string)
        return string.strip()


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
        for utterance in corpus.iter_utterances():
            builder.fit(utterance.text)
        preprocessor = builder.build()
        preprocessor.save()
        print("Preprocessor build saved.")

    # Dataset
    dataset = ChatbotDataset()
    if os.path.exists('build/dataset.json'):
        print("Tokenized dataset found. Using dataset from previous build.")
        with open('build/dataset.json') as f:
            dataset = ChatbotDataset.from_json(f.read())
    else:
        print("Tokenizing data.", end="")
        i = 0
        for conversation in corpus.iter_conversations():
            i = i + 1
            all_utterances = [u for u in conversation.iter_utterances()]
            all_utterances.reverse()
            for (u1, u2) in _pairwise(all_utterances):
                dataset.append(
                    x=preprocessor.prepare(u1.text),
                    y=preprocessor.prepare(u2.text, response=True, add_start=True),
                    z=preprocessor.prepare(u2.text, response=True, add_end=True)
                )
            if i % 500 == 0:
                print(".", end="")
        print("")
        with open('build/dataset.json', 'w', encoding='utf-8') as f:
            f.write(dataset.to_json())
        print("Dataset saved to build/dataset.json")


    return (dataset, preprocessor)
