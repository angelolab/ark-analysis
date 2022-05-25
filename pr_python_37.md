**If you haven't already, please read through our [contributing guidelines](https://ark-analysis.readthedocs.io/en/latest/_rtd/contributing.html) before opening your PR**

**What is the purpose of this PR?**

Closes #506.

Many Libraries have stopped supporting Python 3.6, and more will follow. In order to continue to get support / new features we need to upgrade our Python Version. We'll start with 3.7 and decide if it's worth bumping it up higher.

**How did you implement your changes**

Adjusted the `.travis.yml`, `Dockerfile`, `.readthedocs.yml` and `setup.py`.

**Remaining issues**

## Outdated Packages

```
Package            Version Latest Type
------------------ ------- ------ -----
coverage           5.5     6.4    wheel
coveralls          1.11.1  3.3.1  wheel
cryptography       3.4.8   37.0.2 wheel
ipympl             0.7.0   0.9.1  wheel
jedi               0.17.2  0.18.1 wheel
matplotlib         2.2.5   3.5.2  wheel
mistune            0.8.4   2.0.2  wheel
pandas             0.25.3  1.3.5  wheel
parso              0.7.1   0.8.3  wheel
pluggy             0.13.1  1.0.0  wheel
pytest             5.4.3   7.1.2  wheel
pytest-asyncio     0.17.0  0.18.3 wheel
pytest-cov         2.12.1  3.0.0  wheel
pytest-pycodestyle 2.2.1   2.3.0  sdist
scikit-image       0.16.2  0.19.2 wheel
scikit-learn       0.24.2  1.0.2  wheel
xarray             0.17.0  0.20.2 wheel
```

## Issues

* **Testing:**
  * `pytest` fails to run as `pytest-asyncio` is out of date. 
    * The issue has to do with the `addini` function used in `argparsing.py`. Here is a link to the [issue](https://github.com/pytest-dev/pytest-asyncio/issues/262).
    * Updating `pytest-asyncio` requires updating `pytest`.
  * `visualize_test.py::test_plot_barchart` and `visualize_test.py::test_visualize_patient_population_distribution` fail due to a typing error in `pandas`.
    * We will need to update `pandas` at the very least.
  * `xarray` needs to be updated as a consequence


## Packages to Update
* `pytest`
* `pytest-asyncio`
* `pandas`
* `xarray`