import tensorflow as tf
import yaml


def set_tf_random_seed(seed):
    """Sets the random seed for TensorFlow to ensure reproducibility across runs.

    Args:
        seed (int): The seed value to set for TensorFlow's random number generation.
    """
    print(f"\n\nSet Tensorflow random seed for reproducibility: seed={seed}", flush=True)
    tf.random.set_seed(seed)


def manage_gpu(yaml_config):
    """Manages the GPU configuration based on the YAML configuration file.

    Args:
        yaml_config (str): Path to the YAML configuration file containing the GPU settings.

    Raises:
        ValueError: If an unknown distribution strategy is specified.
    """
    with open(yaml_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    distribute_strategy = config["distribute_strategy"]

    if distribute_strategy == "mirrored":
        set_memory_growth_gpu()
    elif distribute_strategy == "":
        limit_single_gpu()
    else:
        raise ValueError(f"Unknown distribute strategy: {distribute_strategy}")


def set_memory_growth_gpu():
    """Configures TensorFlow to allow GPU memory growth to prevent TensorFlow
    from pre-allocating all available GPU memory.

    Raises:
        RuntimeError: If memory growth settings cannot be applied after GPU initialization.
    """
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
    """Limits TensorFlow to only use the first available GPU.

    Raises:
        RuntimeError: If the GPU visibility cannot be set after initialization.
    """
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


def set_gpu(index):
    """Sets TensorFlow to only use the GPU specified by the index.

    Args:
        index (int): The index of the GPU to use.

    Raises:
        ValueError: If the GPU index is negative or out of range.
        EnvironmentError: If no GPUs are found on the system.
        RuntimeError: If the GPU visibility cannot be set after initialization.
    """
    if index < 0:
        raise ValueError(f"Negative GPU index {index} not allowed.")

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        if len(gpus) <= index:
            raise ValueError(f"GPU index {index} out of range for {len(gpus)} GPUs.")

        print(f"\nLimiting Tensorflow to only use GPU {index}.")

        try:
            tf.config.set_visible_devices(gpus[index], 'GPU')

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        raise EnvironmentError("\nNo GPUs were found.")
