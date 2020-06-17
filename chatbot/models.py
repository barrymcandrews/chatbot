import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Embedding, Dense, TimeDistributed, Dropout, Bidirectional, GRU

import argparse
import os


def Chatbot(vocab_length: int, max_context: int, embeddings=None):
    SharedEmbedding = Embedding(
        input_dim=vocab_length,
        input_length=max_context,
        weights=embeddings,
        mask_zero=True,
        output_dim=256
    )

    units=256

    encoder_input = keras.Input(shape=(max_context,), dtype='int32', name='encoder_input')
    encoder_embedding = SharedEmbedding(encoder_input)

    encoder_1_states  = Bidirectional(GRU(units, return_state=True, return_sequences=True))(encoder_embedding)
    encoder_2_states = Bidirectional(GRU(units, return_state=True, return_sequences=True))(encoder_1_states[0])
    encoder_3_states = GRU(units, return_state=True, return_sequences=True)(encoder_2_states[0])
    encoder_4_states = GRU(units, return_state=True, return_sequences=True)(encoder_3_states[0])
    encoder_5_states = GRU(units, return_state=True, return_sequences=True)(encoder_4_states[0])

    decoder_input = keras.Input(shape=(max_context,), dtype='int32', name='decoder_input')
    decoder_embedding = SharedEmbedding(decoder_input)

    decoder_1_states = GRU(units, return_state=True, return_sequences=True)(decoder_embedding, initial_state=encoder_5_states[1])
    decoder_2_states = GRU(units, return_state=True, return_sequences=True)(decoder_1_states[0])
    decoder_3_states = GRU(units, return_state=True, return_sequences=True)(decoder_2_states[0])
    decoder_4_states = GRU(units, return_state=True, return_sequences=True)(decoder_3_states[0])
    decoder_5_out = GRU(units, return_sequences=True)(decoder_4_states[0])

    decoder_dense = TimeDistributed(Dense(vocab_length, tf.keras.activations.softmax))(decoder_5_out)

    return Model(inputs=[encoder_input, decoder_input], outputs=decoder_dense)
