# Testing

We use [pytest](https://docs.pytest.org/en/latest) to execute the tests. For testing of plot generation, we use the [mpl plugin](https://github.com/matplotlib/pytest-mpl) for pytest. A range of different plots are created, and compared against the baseline images in the `baseline_images` subfolder.

To run the tests, go to the root of this repo directory and run 
```shell
pytest dabest
```


