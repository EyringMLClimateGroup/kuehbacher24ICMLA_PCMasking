import tensorflow as tf


def set_memory_growth_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\nAllowing GPU memory growth.")
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def limit_single_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\nLimiting Tensorflow to only use GPU 0.")
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
