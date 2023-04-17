# Testing

We use [pytest](https://docs.pytest.org/en/latest) to execute the tests. For testing of plot generation, we use the [mpl plugin](https://github.com/matplotlib/pytest-mpl) for pytest. A range of different plots are created, and compared against the baseline images in the `baseline_images` subfolder.

If you have developed a new feature for the package and it is related to modifying original plots or generating new plots, you will need to generate new baseline images. To do so, run 
```shell
pip install -e '.[dev]'
pytest --mpl-generate-path=nbs/tests/baseline_images
```

To run the tests, go to the root of this repo directory and run 
```shell
pytest dabest
```