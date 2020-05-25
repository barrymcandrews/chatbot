import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.embeddings import Embedding

import argparse
import os


def Chatbot(vocab_length: int, max_context: int):
    context_input = keras.Input(shape=(max_context,))
    embedding_layer = Embedding(
        input_dim=vocab_length,
        input_length=max_context,
        output_dim=64
    )(context_input)

    return keras.Model(inputs=context_input, outputs=embedding_layer)
