import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Dropout, Bidirectional

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

    lstm_units=128

    encoder_input = keras.Input(shape=(max_context,), dtype='int32', name='encoder_input')
    encoder_embedding = SharedEmbedding(encoder_input)
    # encoder_embedding_dropout = Dropout(0.3)(encoder_embedding)
    encoder_1_states  = Bidirectional(LSTM(lstm_units, return_state=True, return_sequences=True))(encoder_embedding)
    encoder_2_states = Bidirectional(LSTM(lstm_units, return_state=True, return_sequences=True))(encoder_1_states[0])
    encoder_3_states = LSTM(lstm_units, return_state=True, return_sequences=True)(encoder_2_states[0])
    encoder_4_states = LSTM(lstm_units, return_state=True, return_sequences=True)(encoder_3_states[0])

    decoder_input = keras.Input(shape=(max_context,), dtype='int32', name='decoder_input')
    decoder_embedding = SharedEmbedding(decoder_input)
    # decoder_embedding_dropout = Dropout(0.3)(decoder_embedding)
    decoder_1_states = Bidirectional(LSTM(lstm_units, return_state=True, return_sequences=True))(decoder_embedding, initial_state=encoder_1_states[1:])
    decoder_2_states = Bidirectional(LSTM(lstm_units, return_state=True, return_sequences=True))(decoder_1_states[0], initial_state=encoder_2_states[1:])
    decoder_3_states = Bidirectional(LSTM(lstm_units, return_state=True, return_sequences=True))(decoder_2_states[0], initial_state=encoder_3_states[1:])
    decoder_4_out = LSTM(lstm_units, return_sequences=True)(decoder_3_states[0], initial_state=encoder_4_states[1:])
    # decoder_lstm_dropout = Dropout(0.5)(decoder_lstm)
    decoder_dense = TimeDistributed(Dense(vocab_length, tf.keras.activations.softmax))(decoder_4_out)

    return Model(inputs=[encoder_input, decoder_input], outputs=decoder_dense)
