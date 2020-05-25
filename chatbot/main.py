import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from chatbot.data import load_chatbot_dataset
from chatbot.models import Chatbot
import argparse
import os
import numpy as np
import click

NUM_GPUS = os.environ.get('SM_NUM_GPUS') or 0
MODELS_DIR = os.environ.get('SM_MODEL_DIR') or './models'


@click.group()
@click.option('--model-dir', type=str, default=MODELS_DIR)
@click.pass_context
def cli(ctx, model_dir):
    ctx.obj = model_dir

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
@click.option('--gpu-count', type=int, default=NUM_GPUS)
@click.pass_obj
def train(model_dir, epochs, learning_rate, batch_size, gpu_count):
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
        validation_data=(dataset.testing.x, dataset.testing.y),
        epochs=epochs,
        verbose=1
    )

    chatbot_model.save(model_dir)


if __name__ == '__main__':
    cli()

