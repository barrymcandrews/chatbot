import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model

from chatbot.data import load_movie_dataset, ChatbotData, Dictionary
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
    (dataset, dictionary, max_len) = load_movie_dataset()
    print("Vocabulary Size: " + str(dictionary.size()))
    print("Max Context Length: " + str(max_len))
    chatbot_model = Chatbot(
        dictionary.size(),
        max_len
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
    dictionary = Dictionary.load()
    preprocessor = TextPreprocessor(dictionary, 50)
    while True:
        context = input('you: ')
        context_tokens = tf.convert_to_tensor([preprocessor.prepare(context.split(' '))])

        response = ["<stx>"]
        finished = False
        i = 3
        while not finished:
            processed_response = preprocessor.prepare(response, response=True)
            resp = tf.convert_to_tensor([processed_response])

            result = chatbot_model.predict([context_tokens, resp])

            next_word_index = np.argmax(result[0][i])
            # tops = sorted(range(len(result[0][i])), key = lambda sub: result[0][i][sub])[-10:]
            # print([dictionary.index_to_word[t] for t in tops])

            next_word = dictionary.index_to_word[next_word_index]
            response.append(next_word)

            finished = next_word_index == 0 or next_word_index == 2 or i == 49
            i = i + 1
        print('bot: ' + ' '.join(response))



if __name__ == '__main__':
    cli()

