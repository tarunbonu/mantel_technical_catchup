# utils.py
"""Shared helpers for the Kohonen SOM: config, data, maths and output."""

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import validations

# Config sits next to this file, so the script works from any working directory.
CONFIG_PATH = Path(__file__).parent / 'config.yaml'

logger = logging.getLogger(__name__)


def setup_logging(config):
    """Configure log format and level from the config."""
    logging.basicConfig(
        level=config['logging']['level'],
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def load_config(config_path=CONFIG_PATH):
    """Read the YAML config file, validate it and return it as a dictionary."""
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # safe_load, not load: the config is plain data.
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return validations.validate_config(config)


def resolve_output_dir(config):
    """Return the configured output directory, creating it if missing."""
    output_dir = Path(__file__).parent / config['output']['directory']
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Output directory: %s", output_dir)
    return output_dir


def generate_random_input(n_samples, n_dimensions):
    """Generate n_samples random input vectors with values in [0, 1)."""
    return np.random.random((n_samples, n_dimensions))


def initialise_weights(width, height, n_dimensions):
    """Create the starting weight grid with random values in [0, 1)."""
    return np.random.random((width, height, n_dimensions))


def initial_radius(width, height):
    """Initial neighbourhood radius (sigma_0): half the largest grid dimension."""
    return max(width, height) / 2


def decay_constant(n_iterations, sigma_0):
    """Time constant (lambda) setting how fast the radius and rate decay."""
    return n_iterations / math.log(sigma_0)


def exponential_decay(initial_value, iteration, time_constant):
    """Decay a value exponentially with the current iteration.

    Used for both the radius and the learning rate.
    """
    return initial_value * math.exp(-iteration / time_constant)


def find_bmu(weights, input_vector):
    """Return the (x, y) coordinates of the node closest to input_vector."""
    # Squared distance is enough to find the minimum, so skip the square root.
    # axis=2 sums across the weight vector, leaving one distance per node.
    flat_index = np.argmin(np.sum((weights - input_vector) ** 2, axis=2))
    # argmin works on the flattened grid, so convert back to (x, y).
    return np.unravel_index(flat_index, weights.shape[:2])


def grid_coordinates(width, height):
    """Return the x and y index arrays for the grid.

    Built once per training run and reused every iteration.
    """
    return np.arange(width), np.arange(height)


def neighbourhood_influence(x_coords, y_coords, bmu_x, bmu_y, sigma_t):
    """Influence of the BMU on every node, as a (width, height) array.

    1.0 at the BMU, falling towards 0 with grid distance. This is grid
    distance, not weight-space distance as used by find_bmu.
    """
    # Reshaping to a column and a row makes the two arrays broadcast into the
    # full (width, height) grid of squared distances in one step.
    distance_squared = ((x_coords[:, np.newaxis] - bmu_x) ** 2
                        + (y_coords[np.newaxis, :] - bmu_y) ** 2)
    # Kept squared throughout, so no square root is taken and then undone.
    return np.exp(-distance_squared / (2 * sigma_t ** 2))


def update_weights(weights, input_vector, influence, alpha_t):
    """Move every node towards the input, scaled by its influence.

    Updates in place, so the whole grid is one array operation.
    """
    weights += (alpha_t * influence)[:, :, np.newaxis] * (input_vector - weights)


def save_map(weights, output_path):
    """Save a weight grid as an image, one pixel per node.

    Only meaningful when n_dimensions is 3, since weights are read as RGB.
    """
    validations.validate_image_data(weights)
    plt.imsave(output_path, weights)
    logger.info("Saved map to %s", output_path)
