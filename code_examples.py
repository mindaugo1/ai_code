import tensorflow as tf


def make_dataset(data, vocabulary_size, window_length, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices | (data)
    dataset = dataset.repeat().window(window_length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    dataset = dataset.shuffle(len(data)//100).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    dataset = dataset.map(lambda xs, ys: (tf.one_hot(xs, depth=vocabulary_size), ys))
