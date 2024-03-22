# statfenics

Small set tools for statistical finite element (statFEM) problems, with `fenics`. Contains some utilities that I have found useful, which include:

* A Gaussian process covariance module (`covariance.py`).
* `Fenics` implementation of the Hilbert-GP of [Solin et. al.](https://doi.org/10.1007/s11222-019-09886-w)
* Observation operator creation (`utils.py`).
* Various IO helpers.

The package is very small and is mainly used for my own personal work so it is not necessarily the most robust or well-engineered. Hopefully, however, it is at least useful.

## Installation

At the moment the package is only available from this repo (i.e. not `pypi`), and the installation thus uses a local editable install. This can be done with a one-liner:

```{bash}
pip install --editable git+https://github.com/connor-duffin/statfenics.git#egg=statfenics
```
