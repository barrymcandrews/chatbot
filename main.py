import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.embeddings import Embedding

from data import load_chatbot_dataset
from models import Chatbot
import argparse
import os



def train(args):
    dataset = load_chatbot_dataset()
    chatbot_model = Chatbot(dataset.vocabulary_length, dataset.max_context_length)
    chatbot_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )

    chatbot_model.fit(
        dataset.training.x, dataset.training.y,
        batch_size=args.batch_size,
        validation_data=(dataset.testing.x, dataset.testing.y),
        epochs=args.epochs,
        verbose=1
    )

    chatbot_model.save(args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--gpu-count', type=int, default=os.environ.get('SM_NUM_GPUS') or 0)
    parser.add_argument('--model-dir', type=str, default=os.environ('SM_MODEL_DIR') or './models')
    # parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    train(args)
