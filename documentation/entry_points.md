# Entry points

Recall that all entry points are defined in the `MLproject` file.

In the following command lines, the `--experiment-name` argument is optional.
It could be ignored. Or you could create an experiment, (see [MLflow
cheatsheet](documentation/mlflow_cheatsheet.md)), in which case `expname` should match
your experiment name.

### Load
`mlflow run . -e load --experiment-name=expname -P start_week=180 -P end_week=200`

### Features
`mlflow run . -e features --experiment-name=expname -P start_week=180 -P end_week=200` 

### Validate
`mlflow run . -e validate --experiment-name=expname -P start_week=180 -P end_week=200`

### Train
`mlflow run . -e train --experiment-name=expname -P start_week=180 -P end_week=200`

### Future
`mlflow run . -e future --experiment-name=expname -P next_week=201`

### Predict
`mlflow run . -e predict --experiment-name=expname -P start_week=180 -P end_week=200 -P next_week=201 -P`

<[Précédent](exercises.md) | [Suivant](mlflow_cheatsheet.md)>
