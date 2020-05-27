import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model

from chatbot.data import load_movie_dataset, ChatbotData
from chatbot.models import Chatbot
from chatbot.preprocessor import TextPreprocessor
from sklearn.model_selection import KFold
import argparse
import os
import numpy as np
import click


@click.group()
def cli():
    pass

@cli.command()
def summary():
    chatbot_model = Chatbot(200, 200)
    chatbot_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )
    chatbot_model.summary()


@cli.command()
@click.option('--epochs', type=int, default=10)
@click.option('--learning-rate', type=float, default=0.01)
@click.option('--batch-size', type=int, default=128)
@click.option('--gpu-count', type=int, default=0)
@click.option('--model-dir', type=str, default='build/model')
@click.option('--k-folds', type=int, default=10)
def train(epochs, learning_rate, batch_size, gpu_count, model_dir, k_folds):
    (dataset, preprocessor) = load_movie_dataset()
    print("Vocabulary Size: " + str(preprocessor.get_vocabulary_size()))
    print("Max Context Length: " + str(preprocessor.max_context_length))
    chatbot_model = Chatbot(
        preprocessor.get_vocabulary_size(),
        preprocessor.max_context_length
    )
    chatbot_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )
    chatbot_model.summary()

    kfold = KFold(n_splits=k_folds)

    for train_index, test_index in kfold.split(dataset.x):
        chatbot_model.fit(
            [dataset.x[train_index], dataset.y[train_index]],
            dataset.z[train_index],
            batch_size=batch_size,
            validation_data=(
                [dataset.x[test_index], dataset.y[test_index]],
                dataset.z[test_index]
            ),
            epochs=epochs,
            verbose=1
        )

    chatbot_model.save(model_dir)
    print('Model saved to ' + model_dir)


@cli.command()
@click.option('--build-dir', type=str, default='build')
def chat(build_dir):
    chatbot_model: Model = keras.models.load_model(build_dir + '/model')
    text_preprocessor: TextPreprocessor = TextPreprocessor.load()
    start = text_preprocessor.prepare('<STX>',  is_response=True)
    while True:
        context = input('you: ')
        prepared = text_preprocessor.prepare(context)
        print('input: ' + str(prepared))
        print('start: ' + str(start))

        response = "<stx>"
        finished = False
        i = 0
        while not finished:
            processed_response = text_preprocessor.prepare(response, max_len=5)
            result = chatbot_model.predict([prepared, processed_response])

            predicted_response = [np.argmax(x) for x in result[0]]

            next_word = predicted_response[i]
            print(text_preprocessor.tokenizer.sequences_to_texts([[next_word]]))
            if next_word == 0 or i == 4:
                finished = True

            response = response + ' ' + text_preprocessor.tokenizer.sequences_to_texts([[next_word]])[0]
            print(response)
            i = i + 1



if __name__ == '__main__':
    cli()

