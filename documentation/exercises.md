# Exercises

During this training, you are going to implement the following workflow with MLflow :

* **load:** extract, transform and clean raw data
* **features:** feature engineering
* **validate:** evaluate a model by cross-validation
* **train:** train a model on the whole training set
* **future:** build prediction set
* **predict:** predict on the prediction set

Each step is called an *entry point*.

## Step 1 : become familiar with the pipeline

Exercises : `notebooks/foodcast.ipynb`

In this notebook, you are going to understand and execute the pipeline in "notebook mode", simply to understand the fonctions involved (and already written for you).

To launch the jupyter server: 
* make sure the conda env `foodcast` is activated
* run `jupyter notebook` at the root of the project.

## Step 2 : become familiar with MLflow

Exercises : `notebooks/mlflow_tracking.ipynb`

In this notebook, you are going to understand how to track experiments and how to run the pipeline in "notebook" mode with MLflow. In particular, you will learn how to perform : 
* pipeline reproducibility
* pipeline versioning

## Step 3 : build a multistep workflow with MLflow

In this part, each entry points is a programming exercise. Your goal is to build each entry point sequentially. In particular, you will learn :
* how to cut your pipeline into modular pieces
* how to automatize your workflow

To start training :
* `conda activate foodcast`
* `./start.sh` to remove correction files and run unit tests on `domain` and `infrastructure` only
* Look at the `MLproject` file defining entry points

Your goal is to make each entry point valid, i.e. being actionable by an `mlflow run` command (see cheatsheet below).

This is greatly facilitated by the use of the [click](https://click.palletsprojects.com/en/7.x/) package.

You can also take advantage of `foodcast/application/mlflow_utils.py`.

To test your code :
* `./test.sh entry_point` to run unit tests on the particular entry point you have just coded.

Good luck !

### Exercise 1
* build the `load` entry point by writing a file `foodcast/application/load.py` compliant with`MLproject`
* `./test.sh load`
* `mlflow run . -e load -P start_week=180 -P end_week=200`

### Exercise 2
* build the `features` entry point by writing a file `foodcast/application/features.py` compliant with`MLproject`
* `./test.sh features`
* `mlflow run . -e features -P start_week=180 -P end_week=200`

### Exercise 3
* build the `validate` entry point by writing a file `foodcast/application/validate.py` compliant with`MLproject`
* `./test.sh validate`
* `mlflow run . -e features -P next_week=201`

### Exercise 4
* build the `train` entry point by writing a file `foodcast/application/train.py` compliant with`MLproject`
* `./test.sh train`
* `mlflow run . -e train -P next_week=201`

### Exercise 5
* build the `future` entry point by writing a file `foodcast/application/future.py` compliant with `MLproject`
* `./test.sh future`
* `mlflow run . -e future -P next_week=201`

### Exercise 6
* build the `predict` entry point by writing a file `foodcast/application/predict.py` compliant with`MLproject`
* `./test.sh predict`
* `mlflow run . -e predict -P next_week=201`

And yes, you can rush the exercices by simply typing `./finish.sh`.

<[Précédent](data.md) | [Suivant](entry_points.md)>
