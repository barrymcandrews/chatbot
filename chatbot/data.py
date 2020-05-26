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
class ChatbotDataset():
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __init__(self, x, y, z):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)

@dataclass
class ChatbotData():
    training: ChatbotDataset
    testing: ChatbotDataset
    vocabulary_length: int
    max_context_length: int


def load_chatbot_dataset() -> ChatbotData:
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

    training_dataset = ChatbotDataset(
        x=preprocessor.prepare_texts([d.context for d in training_conversations]),
        y=preprocessor.prepare_texts([d.response for d in training_conversations], is_response=True, add_start=True, max_len=5),
        z=preprocessor.prepare_texts([d.response for d in training_conversations], is_response=True, add_end=True, max_len=5)
    )

    testing_dataset = ChatbotDataset(
        x=preprocessor.prepare_texts([d.context for d in testing_conversations]),
        y=preprocessor.prepare_texts([d.response for d in testing_conversations], is_response=True, add_start=True, max_len=5),
        z=preprocessor.prepare_texts([d.response for d in testing_conversations], is_response=True, add_end=True, max_len=5)
    )


    return ChatbotData(
        training_dataset,
        testing_dataset,
        vocabulary_length=len(vocab) + 3, # needs to be one more than the max size plus STX and ETX
        max_context_length=max_context_length,
    )
