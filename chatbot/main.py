import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model

from chatbot.data import load_dataset, ChatbotData, Dictionary
from chatbot.models import Chatbot
from chatbot.preprocessor import TextPreprocessor, Token
from sklearn.model_selection import KFold
from chatbot.builds import builds as build_command
from chatbot import builds
import argparse
import os
import numpy as np
import click
import datetime
from sklearn.utils import class_weight
from chatbot.util import clean


@click.group()
def cli():
    pass


@cli.command()
def summary():
    chatbot_model = Chatbot(200, 200)
    chatbot_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['sparse_categorical_accuracy']
    )
    chatbot_model.summary()


@cli.command()
@click.option('--epochs', type=int, default=10)
@click.option('--learning-rate', type=float, default=0.00005)
@click.option('--batch-size', type=int, default=256)
@click.option('--build-dir', type=str, default='build')
@click.option('--k-folds', type=int, default=10)
@click.option('--upload', is_flag=True)
@click.option('--continue', '-c', 'continue_', is_flag=True)
@click.option('--base-build', type=str, default=None)
@click.option('--prep', is_flag=True)
def train(epochs, learning_rate, batch_size, build_dir, k_folds, upload, continue_, base_build, prep):
    if base_build:
        builds.download_build(base_build)

    (dataset, dictionary, max_len) = load_dataset()
    print("Vocabulary Size: " + str(dictionary.size()))
    print("Max Context Length: " + str(max_len))
    print("Dataset Length: " + str(len(dataset.x)))

    chatbot_model = None
    if continue_:
        chatbot_model = keras.models.load_model(build_dir + '/model')
    else:
        chatbot_model = Chatbot(
            dictionary.size(),
            max_len,
            embeddings=None
        )
        chatbot_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(reduction='none'),
            weighted_metrics=['sparse_categorical_accuracy']
        )
    chatbot_model.summary()


    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Print out dataset for manual inspection
    with open('out.txt', 'w', encoding='utf-8') as f:
        for n in range(len(dataset.x)):
            w_x = [dictionary.index_to_word[x_i] for x_i in dataset.x[n]]
            w_y = [dictionary.index_to_word[y_i] for y_i in dataset.y[n]]
            f.write("\n%s\n%s\n" %  (" ".join(w_x).strip(), " ".join(w_y).strip()))

    if prep:
        exit(0)

    for e in range(epochs):
        kfold = KFold(n_splits=k_folds, shuffle=True)
        k = 0
        for train_index, test_index in kfold.split(dataset.x):

            classes = np.unique(dataset.z)
            # weights_array = class_weight.compute_class_weight('balanced', classes=classes, y=np.ravel(dataset.z))
            weights = {i: 1 for i in range(dictionary.size())}
            # weights.update({classes[i]: w for i, w in enumerate(weights_array)})
            weights.update({Token.ETX: 1e-6})

            print("Epoch: %d.%d" % (e, k))
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
                epochs=10,
            )
            k = k + 1
            chatbot_model.save(build_dir + '/model')
            print('Model saved to ' + build_dir + '/model')

    if upload:
        builds.upload_build()


@cli.command()
@click.option('--build-dir', type=str, default='build')
def chat(build_dir):
    chatbot_model: Model = keras.models.load_model(build_dir + '/model')
    dictionary = Dictionary.load()
    MAX_LEN = 30
    preprocessor = TextPreprocessor(dictionary, MAX_LEN)

    while True:
        context = input('you: ')
        print("cleaned: " + str(clean(context)))
        print("tokens: " + str(preprocessor.prepare(clean(context))))
        context_tokens = tf.convert_to_tensor([preprocessor.prepare(clean(context))])

        response = ['<u0>']
        for i in range(1, MAX_LEN):
            processed_response = preprocessor.prepare(response, response=True, add_end=False)
            resp = tf.convert_to_tensor([processed_response])

            result = chatbot_model.predict([context_tokens, resp])
            predictions = np.argmax(result[0], axis=1)

            pred_words = [dictionary.index_to_word[x] for x in predictions]
            print(pred_words)

            next_word_index = np.argmax(result[0][i])
            tops = np.argpartition(result[0][i], -5)[-5:]
            top_probs = [result[0][i][n] for n in tops]

            top_words = [dictionary.index_to_word[t] for t in tops]
            print(sorted(zip(top_probs, top_words), reverse=True))

            next_word = dictionary.index_to_word[next_word_index]
            response.append(next_word)

            if next_word_index == Token.ETX:
                break

        print('bot: ' + ' '.join(response))


@cli.command()
def clean_text():
    dictionary = Dictionary.load()
    MAX_LEN = 30
    preprocessor = TextPreprocessor(dictionary, MAX_LEN)

    while True:
        context = input('you: ')
        c = clean(context)
        print("cleaned: " + str(c))
        print("tokens: " + str(preprocessor.prepare(clean(context))))


cli.add_command(build_command)

if __name__ == '__main__':
    cli()

