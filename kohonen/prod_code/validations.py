# validations.py
"""Input and output validation for the Kohonen SOM.

Checks run before training starts and after it finishes, so bad arguments fail
with a clear message instead of surfacing later as a numpy shape error.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Required keys in config.yaml, as (section, key) pairs.
REQUIRED_CONFIG_KEYS = [
    ('som', 'initial_learning_rate'),
    ('som', 'n_dimensions'),
    ('output', 'directory'),
    ('logging', 'level'),
    ('logging', 'log_every'),
]

# Smallest grid that still gives a usable decay constant.
# sigma_0 is max(width, height) / 2, and decay_constant divides by log(sigma_0),
# so sigma_0 must be greater than 1, meaning the larger side needs at least 3 nodes.
MIN_GRID_SIDE = 3


def validate_config(config):
    """Check the config has every required key with a usable value."""
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config).__name__}")

    for section, key in REQUIRED_CONFIG_KEYS:
        if section not in config:
            raise ValueError(f"Config is missing the '{section}' section")
        if key not in config[section]:
            raise ValueError(f"Config is missing '{section}.{key}'")

    learning_rate = config['som']['initial_learning_rate']
    if not isinstance(learning_rate, (int, float)) or not 0 < learning_rate <= 1:
        raise ValueError(
            f"som.initial_learning_rate must be between 0 and 1, got {learning_rate}"
        )

    n_dimensions = config['som']['n_dimensions']
    if not isinstance(n_dimensions, int) or n_dimensions < 1:
        raise ValueError(
            f"som.n_dimensions must be a positive integer, got {n_dimensions}"
        )

    log_every = config['logging']['log_every']
    if not isinstance(log_every, int) or log_every < 1:
        raise ValueError(
            f"logging.log_every must be a positive integer, got {log_every}"
        )

    logger.debug("Config validated")
    return config


def validate_training_inputs(input_data, n_max_iterations, width, height,
                             alpha_0, n_dimensions):
    """Check the arguments to train() before any work starts."""
    input_data = np.asarray(input_data)

    if input_data.ndim != 2:
        raise ValueError(
            f"input_data must be 2D (n_samples, n_dimensions), "
            f"got {input_data.ndim}D with shape {input_data.shape}"
        )
    if input_data.shape[0] == 0:
        raise ValueError("input_data is empty")
    if input_data.shape[1] != n_dimensions:
        raise ValueError(
            f"input_data has {input_data.shape[1]} dimensions "
            f"but n_dimensions is {n_dimensions}"
        )
    if not np.isfinite(input_data).all():
        raise ValueError("input_data contains NaN or infinite values")

    for name, value in (('width', width), ('height', height)):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value}")

    if max(width, height) < MIN_GRID_SIDE:
        raise ValueError(
            f"The larger grid side must be at least {MIN_GRID_SIDE} nodes "
            f"for the decay constant to be valid, got {width}x{height}"
        )

    if not isinstance(n_max_iterations, int) or n_max_iterations < 1:
        raise ValueError(
            f"n_max_iterations must be a positive integer, got {n_max_iterations}"
        )

    if not isinstance(alpha_0, (int, float)) or not 0 < alpha_0 <= 1:
        raise ValueError(f"alpha_0 must be between 0 and 1, got {alpha_0}")

    logger.debug("Training inputs validated")
    return input_data


def validate_trained_weights(weights, width, height, n_dimensions):
    """Check the trained grid before it is used or saved."""
    if weights.shape != (width, height, n_dimensions):
        raise ValueError(
            f"Trained weights have shape {weights.shape}, "
            f"expected {(width, height, n_dimensions)}"
        )
    # NaN or inf here usually means the decay schedule diverged.
    if not np.isfinite(weights).all():
        raise ValueError("Trained weights contain NaN or infinite values")

    logger.debug("Trained weights validated")
    return weights


def validate_image_data(weights):
    """Check a weight grid can be saved as a meaningful image."""
    if weights.ndim != 3 or weights.shape[2] != 3:
        raise ValueError(
            f"Image data must have shape (width, height, 3), got {weights.shape}"
        )
    # imsave silently clips floats outside [0, 1], which hides bad training.
    if weights.min() < 0 or weights.max() > 1:
        raise ValueError(
            f"Image data must be in [0, 1], got range "
            f"[{weights.min():.3f}, {weights.max():.3f}]"
        )

    logger.debug("Image data validated")
    return weights
