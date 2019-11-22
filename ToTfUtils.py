import tensorflow as tf


class ToTfUtilities:

    @staticmethod
    def create_tf_dataset_from_data_bundle_obj(data_bundle):
        x_dataset = tf.data.Dataset.from_tensor_slices(data_bundle.x)
        y_dataset = tf.data.Dataset.from_tensor_slices(data_bundle.y)
        return tf.data.Dataset.zip((x_dataset, y_dataset))

    @staticmethod
    def make_one_hot_tf_dataset_from_sequence_data(data, vocabulary_size, window_length, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.repeat().window(window_length, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_length))
        dataset = dataset.shuffle(len(data)//100).batch(batch_size)
        dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        dataset = dataset.map(lambda xs, ys: (tf.one_hot(xs, depth=vocabulary_size), ys))
