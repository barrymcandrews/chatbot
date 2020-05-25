import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model

from chatbot.data import load_chatbot_dataset
from chatbot.models import Chatbot
from chatbot.preprocessor import TextPreprocessor
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
def train(epochs, learning_rate, batch_size, gpu_count, model_dir):
    dataset = load_chatbot_dataset()
    chatbot_model = Chatbot(dataset.vocabulary_length, dataset.max_context_length)
    chatbot_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )
    chatbot_model.summary()

    chatbot_model.fit(
        [dataset.training.x, dataset.training.y],
        dataset.training.y,
        batch_size=batch_size,
        validation_data=([dataset.testing.x, dataset.testing.y], dataset.testing.y),
        epochs=epochs,
        verbose=1
    )

    chatbot_model.save(model_dir)


@cli.command()
@click.option('--build-dir', type=str, default='build')
def chat(model_dir):
    chatbot_model: Model = keras.models.load_model(model_dir)
    text_preprocessor: TextPreprocessor = TextPreprocessor.load()
    while True:
        context = input('you: ')
        prepared = text_preprocessor.prepare(context)
        chatbot_model.predict([prepared])


if __name__ == '__main__':
    cli()

