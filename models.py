import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.embeddings import Embedding

from data import load_babi_data
import argparse
import os


def Chatbot(vocab_length: int, max_context: int):
    inputs = keras.Input(shape=(None,))
    embedding_layer = Embedding(
        input_dim=vocab_length,
        input_length=max_context,
        output_dim=64
    )(inputs)

    return keras.Model(inputs=inputs, outputs=embedding_layer)
