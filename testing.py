from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time

import tensorflow as tf
import numpy as np

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def create_dataset(text_as_int, seq_length, examples_per_epoch):
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    return sequences.map(split_input_target)

def create_training_batches(dataset, batch_size):
    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000
    return dataset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def print_example(dataset, idx2char):
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)

        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

        print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
        print()
        print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

        example_batch_loss = loss(target_example_batch, example_batch_predictions)
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("scalar_loss:      ", example_batch_loss.numpy().mean())


path_to_file = tf.keras.utils.get_file(
        'shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Unique characters, create mappings
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


if __name__ == '__main__':
    tf.enable_eager_execution()

    print('Length of text: {} characters.'.format(len(text)))
    print('Text contains {} unique characters.'.format(len(vocab)))

    text_as_int = np.array([char2idx[c] for c in text])
    dataset = create_dataset(text_as_int, 100, len(text))

    BATCH_SIZE = 64
    dataset = create_training_batches(dataset, BATCH_SIZE)

    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
    
    print_example(dataset, idx2char)

    model.compile(optimizer='adam', loss=loss)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "chpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS=10
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # tf.train.latest_checkpoint(checkpoint_dir)
