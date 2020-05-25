import os
import re
from functools import reduce
from dataclasses import dataclass
from typing import List, Set
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences


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
                    response=segment[1]
                )
                interaction.context.extend(words)
                interactions.append(interaction)

    return interactions


@dataclass
class DatasetPair():
    x: tf.data.Dataset
    y: tf.data.Dataset

    def __init__(self, x, y):
        self.x = tf.data.Dataset.from_tensor_slices(x)
        self.y = tf.data.Dataset.from_tensor_slices(y)

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
    print("max context length: " + str(max_context_length))

    vocab = (set()
        .union({x for y in training_conversations for x in y.context })
        .union({x for y in training_conversations for x in y.response })
        .union({x for y in testing_conversations for x in y.context })
        .union({x for y in testing_conversations for x in y.response }))

    tokenizer = Tokenizer(filters = [])
    tokenizer.fit_on_texts(vocab)

    def tokenize(texts):
        return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_context_length)

    training_dataset_pair = DatasetPair(
        x=tokenize([d.context for d in training_conversations]),
        y=tokenize([d.response for d in training_conversations])
    )

    testing_dataset_pair = DatasetPair(
        x=tokenize([d.context for d in testing_conversations]),
        y=tokenize([d.response for d in testing_conversations])
    )


    return ChatbotDataset(
        training_dataset_pair,
        testing_dataset_pair,
        vocabulary_length=len(vocab) + 1,
        max_context_length=max_context_length,
    )



