import os
import re
from functools import reduce
from dataclasses import dataclass
from typing import List, Set
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from chatbot.preprocessor import TextPreprocessor

DATA_DIR = './data/bAbI'

@dataclass
class Interaction():
    context: List[str]
    response: List[str]

def flatten(list):
    return [x for y in list for x in y]


def read_babi_file(filename: str) -> List[Interaction]:
    interactions = []

    statements = []

    with open(DATA_DIR + '/' + filename) as f:
        for line in f:
            # separate tokens into their own words
            line = line.lower()
            line = re.sub(r'(\n)', '', line)
            line = re.sub(r'(\.)', ' .', line)
            line = re.sub(r'(\? )', '?', line)
            line = re.sub(r'(\?)', ' ?', line)

            segment = line.split('\t')
            words = segment[0].split(' ')
            index = words.pop(0)
            if index == '1' and len(interactions) > 0:
                statements = []

            if len(segment) == 1:
                statements.append(words)
            else:
                interaction = Interaction(
                    context=flatten(statements),
                    response=segment[1].split(' ')
                )
                interaction.context.extend(words)
                interactions.append(interaction)

    return interactions


@dataclass
class DatasetPair():
    x: np.ndarray
    y: np.ndarray

    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

@dataclass
class ChatbotDataset():
    training: DatasetPair
    testing: DatasetPair
    vocabulary_length: int
    max_context_length: int


def load_chatbot_dataset() -> ChatbotDataset:
    dataset_names = {re.sub(r'(_test.txt)|(_train.txt)$', '', x) for x in os.listdir(DATA_DIR)}

    training_conversations: List[Interaction] = []
    testing_conversations: List[Interaction] = []

    for dataset_name in dataset_names:
        training_conversations.extend(read_babi_file(dataset_name + '_train.txt'))
        testing_conversations.extend(read_babi_file(dataset_name + '_test.txt'))

    max_context_length = max(
        max([len(x.context) for x in training_conversations]),
        max([len(x.context) for x in testing_conversations])
    )

    vocab = (set()
        .union({x for y in training_conversations for x in y.context })
        .union({x for y in training_conversations for x in y.response })
        .union({x for y in testing_conversations for x in y.context })
        .union({x for y in testing_conversations for x in y.response }))

    preprocessor = TextPreprocessor(vocab, max_context_length)
    preprocessor.save()

    training_dataset_pair = DatasetPair(
        x=preprocessor.prepare_texts([d.context for d in training_conversations]),
        y=preprocessor.prepare_texts([d.response for d in training_conversations], add_tokens=True)
    )

    testing_dataset_pair = DatasetPair(
        x=preprocessor.prepare_texts([d.context for d in testing_conversations]),
        y=preprocessor.prepare_texts([d.response for d in testing_conversations], add_tokens=True)
    )


    return ChatbotDataset(
        training_dataset_pair,
        testing_dataset_pair,
        vocabulary_length=len(vocab) + 3, # needs to be one more than the max size plus STX and ETX
        max_context_length=max_context_length,
    )
