
# Differences with code template

* `Makefile` is simplified, the `--cov-branch` is added to coverage option
* `foodcast/settings/dev.py` defines a `TEST_DATA_DIR` path
* no `init.sh`, simply type  `conda env install -f conda.yaml`
* no `activate.sh`, simply type `conda activate foodcast`
* no `requirements.txt` nor `requirements.dev.txt`, replaced by `conda.yaml`
* additional `MLproject` file, which is a kind of MLflow Makefile
* added `mlruns` to `.gitignore`

<[Précédent](mlflow_cheatsheet.md) |
