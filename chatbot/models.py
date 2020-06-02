import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed

import argparse
import os


def Chatbot(vocab_length: int, max_context: int, embeddings=None):
    SharedEmbedding = Embedding(
        input_dim=vocab_length,
        input_length=max_context,
        weights=embeddings,
        mask_zero=True,
        output_dim=300
    )

    encoder_input = keras.Input(shape=(max_context,), dtype='int32')
    encoder_embedding = SharedEmbedding(encoder_input)
    _, state_h, state_c = LSTM(300, return_state=True)(encoder_embedding)

    decoder_input = keras.Input(shape=(max_context,), dtype='int32')
    decoder_embedding = SharedEmbedding(decoder_input)
    decoder_lstm, _, _ = LSTM(300, return_state=True, return_sequences=True)(decoder_embedding, initial_state=[state_h, state_c])
    # decoder_dense = Dense(vocab_length, tf.keras.activations.softmax)(decoder_lstm)
    # middle = Dense(300, tf.keras.activations.relu)(decoder_lstm)
    decoder_dense = TimeDistributed(Dense(vocab_length, tf.keras.activations.softmax))(decoder_lstm)

    return Model(inputs=[encoder_input, decoder_input], outputs=decoder_dense)
