# Docs for DABEST-Python

## Required Packages

```shell
pip install --upgrade sphinx sphinx-autobuild
```

[Sphinx](http://www.sphinx-doc.org/en/master/index.html) is the main documentation package; [sphinx-autobuild](https://github.com/GaretJax/sphinx-autobuild) is used for hot-load development.

## Running hot-load development

After cloning the repo, run

```shell
sphinx-autobuild docs/source docs/build/_html
```

## Adding custom CSS

See the official [docs](https://docs.readthedocs.io/en/latest/guides/adding-custom-css.html), as well as this Stack Overflow [thread](https://stackoverflow.com/questions/23462494/how-to-add-custom-css-file-to-sphinx) to assign a custom CSS to the Sphinx template.

1. Place the following line into `source/_templates/layout.html`:

```css
{% set css_files = css_files + ['_static/css/alabaster-custom.css'] %}
```

2. If hot-loading doesn't reload the custom CSS, simply copy your custom CSS into the folder `_html/_static/css`.
