import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model

from chatbot.data import load_movie_dataset, ChatbotData, Dictionary
from chatbot.models import Chatbot
from chatbot.preprocessor import TextPreprocessor, Token
from sklearn.model_selection import KFold
import argparse
import os
import numpy as np
import click
import datetime
from sklearn.utils import class_weight


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
@click.option('--build-dir', type=str, default='build')
@click.option('--k-folds', type=int, default=10)
@click.option('--cloud-dir', type=str, default=None)
def train(epochs, learning_rate, batch_size, build_dir, k_folds, job_dir):
    (dataset, dictionary, max_len) = load_movie_dataset()
    print("Vocabulary Size: " + str(dictionary.size()))
    print("Max Context Length: " + str(max_len))
    print("Dataset Length: " + str(len(dataset.x)))
    chatbot_model = Chatbot(
        dictionary.size(),
        max_len
    )
    chatbot_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=['sparse_categorical_accuracy']
    )
    chatbot_model.summary()


    for e in range(epochs):
        kfold = KFold(n_splits=k_folds)
        for train_index, test_index in kfold.split(dataset.x):

            # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            classes = np.unique(dataset.z)
            weights_array = class_weight.compute_class_weight('balanced', classes=classes, y=np.ravel(dataset.z))
            weights = {i: 1 for i in range(dictionary.size())}
            # weights.update({classes[i]: w for i, w in enumerate(weights_array)})

            print("Epoch: " + str(e))
            chatbot_model.fit(
                [dataset.x[train_index], dataset.y[train_index]],
                dataset.z[train_index],
                batch_size=batch_size,
                validation_data=(
                    [dataset.x[test_index], dataset.y[test_index]],
                    dataset.z[test_index]
                ),
                # callbacks=[tensorboard_callback],
                class_weight=weights,
            )

    chatbot_model.save(build_dir + '/model')
    print('Model saved to ' + build_dir + '/model')


@cli.command()
@click.option('--build-dir', type=str, default='build')
def chat(build_dir):
    chatbot_model: Model = keras.models.load_model(build_dir + '/model')
    dictionary = Dictionary.load()
    MAX_LEN = 20
    preprocessor = TextPreprocessor(dictionary, MAX_LEN)

    while True:
        context = input('you: ')
        context_tokens = tf.convert_to_tensor([preprocessor.prepare(context.split(' '))])

        response = ["<stx>"]
        for i in range(MAX_LEN):
            processed_response = preprocessor.prepare(response, response=True)
            resp = tf.convert_to_tensor([processed_response])

            result = chatbot_model.predict([context_tokens, resp])
            # predictions = np.argmax(result[0], axis=1)
            # print("\ni = " + str(i))
            # print(predictions)
            # print([dictionary.index_to_word[x] for x in predictions])

            next_word_index = np.argmax(result[0][i])
            # tops = np.argpartition(-result[0][i], 5)
            # top_probs = [result[0][i][n] for n in tops]

            # print([dictionary.index_to_word[t] for t in tops])
            # print(top_probs)

            next_word = dictionary.index_to_word[next_word_index]
            response.append(next_word)

            if next_word_index == Token.ETX:
                break

        print('bot: ' + ' '.join(response))



if __name__ == '__main__':
    cli()

