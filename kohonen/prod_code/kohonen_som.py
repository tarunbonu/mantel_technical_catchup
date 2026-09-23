# kohonen_som.py
"""Kohonen Self-Organising Map training.

A SOM learns a 2D grid of weight vectors that approximates a higher-dimensional
input space. Each iteration finds the node closest to an input vector (the Best
Matching Unit) and pulls it and its neighbours towards that input. The radius
and learning rate shrink over time, so the map organises coarsely then refines.
"""

import logging
import time

import utils
import validations

logger = logging.getLogger(__name__)


def train(input_data, n_max_iterations, width, height, alpha_0, n_dimensions,
          log_every=10):
    """Train a SOM and return the trained weight grid.

    Args:
        input_data: array of shape (n_samples, n_dimensions).
        n_max_iterations: number of passes over the whole input set.
        width, height: grid dimensions of the map.
        alpha_0: initial learning rate.
        n_dimensions: length of each weight vector.
        log_every: log progress every this many iterations.

    Returns:
        Array of shape (width, height, n_dimensions).
    """
    # Fail early with a clear message rather than deep inside the loop.
    input_data = validations.validate_training_inputs(
        input_data, n_max_iterations, width, height, alpha_0, n_dimensions
    )

    logger.info(
        "Training start: %dx%d grid, %d iterations, %d inputs, alpha_0=%s",
        width, height, n_max_iterations, len(input_data), alpha_0,
    )
    start = time.perf_counter()

    sigma_0 = utils.initial_radius(width, height)
    time_constant = utils.decay_constant(n_max_iterations, sigma_0)
    weights = utils.initialise_weights(width, height, n_dimensions)
    # Grid coordinates never change, so build them once rather than per iteration.
    x_coords, y_coords = utils.grid_coordinates(width, height)
    logger.debug("sigma_0=%.3f, time_constant=%.3f", sigma_0, time_constant)

    for iteration in range(n_max_iterations):
        # Radius and learning rate share one decay schedule.
        sigma_t = utils.exponential_decay(sigma_0, iteration, time_constant)
        alpha_t = utils.exponential_decay(alpha_0, iteration, time_constant)

        for input_vector in input_data:
            # Find the node closest to this input.
            bmu_x, bmu_y = utils.find_bmu(weights, input_vector)

            # Influence and update are computed for the whole grid at once.
            influence = utils.neighbourhood_influence(
                x_coords, y_coords, bmu_x, bmu_y, sigma_t
            )
            utils.update_weights(weights, input_vector, influence, alpha_t)

        # Progress, with the decayed values so a bad decay schedule is visible early.
        if (iteration + 1) % log_every == 0 or iteration == n_max_iterations - 1:
            elapsed = time.perf_counter() - start
            logger.info(
                "Iteration %d/%d (%.0f%%): sigma=%.3f, alpha=%.4f, elapsed=%.2fs",
                iteration + 1, n_max_iterations,
                100 * (iteration + 1) / n_max_iterations,
                sigma_t, alpha_t, elapsed,
            )

    validations.validate_trained_weights(weights, width, height, n_dimensions)
    logger.info(
        "Training done: %dx%d grid in %.2fs",
        width, height, time.perf_counter() - start,
    )
    return weights


if __name__ == '__main__':
    # Constants, logging and output location come from config.yaml.
    config = utils.load_config()
    utils.setup_logging(config)
    alpha_0 = config['som']['initial_learning_rate']
    n_dimensions = config['som']['n_dimensions']
    log_every = config['logging']['log_every']
    output_dir = utils.resolve_output_dir(config)

    try:
        # Small map: 10x10 grid, 100 iterations.
        input_data = utils.generate_random_input(10, n_dimensions)
        image_data = train(input_data, 100, 10, 10, alpha_0, n_dimensions, log_every)
        utils.save_map(image_data, output_dir / '100.png')

        # Large map: 100x100 grid, 1000 iterations.
        input_data = utils.generate_random_input(10, n_dimensions)
        image_data = train(input_data, 1000, 100, 100, alpha_0, n_dimensions, log_every)
        utils.save_map(image_data, output_dir / '1000.png')
    except Exception:
        # exception() logs the full traceback, so failures are debuggable from the log.
        logger.exception("Training failed")
        raise
