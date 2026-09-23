# Kohonen Self-Organising Map (SOM)

Production-style implementation of a Kohonen Self-Organising Map. This directory is
intended to reflect how the algorithm would be written, structured and maintained in a
real codebase rather than as a notebook experiment: clear module boundaries, configuration
kept out of the code, a pinned environment, and code that a new team member can read and
extend without reverse-engineering it. What is still missing is listed at the end, rather
than left implied.

## Contents

| File | Purpose |
|------|---------|
| `kohonen_som.py` | Core SOM training logic and an example entry point |
| `utils.py` | Shared helpers: config loading, data generation, SOM maths, saving output |
| `validations.py` | Input and output validation, raised as clear errors before training starts |
| `config.yaml` | Configuration: learning rate, dimensions, output directory, logging |
| `requirements.txt` | Pinned dependencies |
| `README.md` | This file |

## What is a Kohonen SOM?

A Self-Organising Map is an unsupervised neural network, introduced by Teuvo Kohonen,
that learns a low-dimensional (usually 2D) grid of nodes to represent a higher-dimensional
input space. Each node holds a weight vector with the same dimensionality as the input
data. Training pulls those weight vectors towards the data so that, at the end, nearby
nodes on the grid hold similar weight vectors and the grid as a whole approximates the
shape of the input distribution.

Training loop, for each iteration `t` and each input vector `v`:

1. **Find the Best Matching Unit (BMU):** the node whose weight vector is closest to `v`
   by Euclidean distance.
2. **Compute the neighbourhood:** every node gets an influence `θ` that decays with its
   grid distance `d` from the BMU, using a Gaussian:
   `θ(t) = exp(-d² / (2 σ(t)²))`.
3. **Update weights:** each node moves towards `v` in proportion to its influence and the
   current learning rate: `w ← w + α(t) · θ(t) · (v − w)`.

Both the neighbourhood radius `σ(t)` and the learning rate `α(t)` shrink over time with
exponential decay, so early iterations organise the map coarsely and later iterations
fine-tune it locally:

```
σ(t) = σ0 · exp(-t / λ)        σ0 = max(width, height) / 2
α(t) = α0 · exp(-t / λ)        λ  = n_iterations / ln(σ0)
```

## Why use it

**Dimensionality reduction.** A SOM maps every input vector to a single grid coordinate
(its BMU). A 3-dimensional colour, or a 50-dimensional customer profile, becomes a point on
a 2D grid. Unlike PCA the mapping is non-linear, and unlike t-SNE or UMAP the result is a
fixed, trained model that can place new, unseen points without retraining. The grid is
also directly plottable, which makes it a practical tool for visualising structure in
high-dimensional data.

**Unsupervised learning.** No labels are needed. The map learns purely from the geometry
of the inputs, so it is useful for:

- **Clustering:** regions of the grid whose nodes have similar weights correspond to
  clusters in the data. A U-matrix (distance between neighbouring nodes) makes the
  cluster boundaries visible.
- **Exploratory analysis:** seeing which inputs land near each other, and which parts of
  the grid are densely or sparsely populated.
- **Anomaly detection:** inputs whose distance to their BMU is unusually large do not fit
  the learned structure well.
- **Vector quantisation:** the trained weight vectors form a compact codebook that
  summarises the dataset.

The example in this repository trains on random RGB colours. The output image is the
trained weight grid rendered as pixels: similar colours end up next to each other, which
is the SOM's topology-preserving property made visible.

## Setup

Requires Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To add a dependency, install it into the venv and then add it with a pinned version
to `requirements.txt`, so that the file stays the single source of truth for what the
project needs.

## Usage

```bash
python kohonen_som.py
```

This trains two example maps (10x10 over 100 iterations, and 100x100 over 1000
iterations) and writes each result as a PNG into the configured output directory.

Algorithm constants, the output location and logging behaviour come from `config.yaml`.
Grid size and iteration count are still passed as arguments at the call site in
`__main__`; moving those into the config as named run profiles is a sensible next step.

## Design principles for this directory

- **Configuration lives in `config.yaml`, not in code.** Changing the learning rate or the
  input dimensionality should never require editing Python.
- **Vectorised numerics.** Node updates are computed with NumPy array operations rather
  than per-node Python loops, so large grids remain tractable. On a 100x100 grid over 1000
  iterations this is the difference between 5m 54s and roughly 4.8s.
- **Small, single-purpose functions** with docstrings, so each step (BMU lookup,
  neighbourhood, weight update) can be tested and reasoned about on its own.
- **Validate at the boundaries.** Arguments are checked before training starts and the
  trained grid is checked before it is returned or saved, so bad input fails with a clear
  message instead of surfacing later as a NumPy shape error or a plausible but wrong map.

## Not done yet

Called out explicitly rather than left for a reader to discover:

- **No unit tests.** The properties worth asserting are known (influence is 1.0 at the BMU
  and falls with distance; decay is monotonic; the returned grid has the expected shape),
  but the suite is not written. This is the next piece of work.
- **Runs are not seeded.** Weight initialisation and input generation both call
  `np.random` without a seed, so no two runs produce an identical map. A seed belongs in
  `config.yaml`, and is a prerequisite for meaningful tests.
- **No type hints.** Shapes and types are documented in the docstrings but not enforced.
- **No container.** Dependencies are pinned, which is what makes adding a Dockerfile a
  small step rather than a project.
