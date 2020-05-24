import os
import re
from functools import reduce
from dataclasses import dataclass
from typing import List, Set
from keras.preprocessing.text import Tokenizer
import tensorflow as tf


DATA_DIR = './data/bAbI'

@dataclass
class Data():
    context: List[str]
    response: List[str]

Conversation = List[Data]


def read_babi_file(filename: str) -> List[Conversation]:
    conversations = []
    conversation = []
    data: Data = Data([], [])

    with open(DATA_DIR + '/' + filename) as f:
        for line in f:

            # separate tokens into their own words
            line = line.lower()
            line = re.sub(r'(\n)', '', line)
            line = re.sub(r'(\.)', ' .', line)
            line = re.sub(r'(\? )', '?', line)
            line = re.sub(r'(\?)', ' ?', line)

            statements = line.split('\t')
            words = statements[0].split(' ')
            index = words.pop(0)
            if index == '1' and len(conversation) > 0:
                conversations.append(conversation)
                conversation = []

            data.context.extend(words)

            if len(statements) > 1:
                data.response = statements[1].split(' ')
                conversation.extend([data])
                data = Data([], [])
    return conversations


def flatten_convo(convo: Conversation) -> Data:
    return convo[0]


@dataclass
class DatasetPair():
    x: tf.data.Dataset
    y: tf.data.Dataset

@dataclass
class ChatbotDataset():
    training: DatasetPair
    testing: DatasetPair
    vocabulary_length: int
    max_context_length: int


def load_babi_data() -> ChatbotDataset:
    pass


def load_chatbot_dataset() -> ChatbotDataset:
    dataset_names = {re.sub(r'(_test.txt)|(_train.txt)$', '', x) for x in os.listdir(DATA_DIR)}

    training_conversations = []
    testing_conversations = []

    for dataset_name in dataset_names:
        training_conversations.extend(read_babi_file(dataset_name + '_train.txt'))
        testing_conversations.extend(read_babi_file(dataset_name + '_test.txt'))

    training_conversations = [ flatten_convo(c) for c in training_conversations ]
    testing_conversations = [ flatten_convo(c) for c in testing_conversations ]

    training_dataset_pair = DatasetPair(
        x=[d.context for d in training_conversations],
        y=[d.response for d in training_conversations]
    )

    testing_dataset_pair = DatasetPair(
        x=[d.context for d in testing_conversations],
        y=[d.response for d in testing_conversations]
    )




    return ChatbotDataset(
        training_dataset_pair,
        testing_dataset_pair
    )



