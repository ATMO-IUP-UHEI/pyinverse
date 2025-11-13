# pyinverse

Solve various inverse problems!

## Installation

1. Install poetry as a package manager (<https://python-poetry.org/docs/#installation>)
2. Download the source code from GitHub:

    ```bash
    $ git clone https://github.com/ATMO-IUP-UHEI/pyinverse.git
    ```
3. Install `numba` and `llvmlite` either manually or create a conda environment using the provided file:
    ```bash
    $ conda env create -f conda-environment.yaml
    ```
4. Open the directory and install with poetry:

    ```bash
    cd pyinverse
    poetry install
    ```

<!-- ```bash
$ pip install pyinverse
``` -->

## Usage

```python
import numpy as np
from pyinverse.loss import Bayesian
from pyinverse.solver import BayesianAnalytical

# Define your inverse problem
y = np.array([1.2, 2.3, 1.8])           # measurements
K = np.array([[1, 0], [0, 1], [1, 1]])  # forward model
x_prior = np.array([1.0, 1.0])          # prior state
cov_prior = np.eye(2)                    # prior covariance
cov_y = 0.1 * np.eye(3)                  # measurement covariance

# Set up Bayesian inversion
loss = Bayesian(y, cov_y, K, x_prior, cov_prior)
solver = BayesianAnalytical(loss)

# Solve for posterior state
x_post = solver.x_posterior
cov_post = solver.cov_posterior
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pyinverse` was created by Robert Maiwald and Christopher Lüken-Winkels. It is licensed under the terms of the MIT license.

## Credits

`pyinverse` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

