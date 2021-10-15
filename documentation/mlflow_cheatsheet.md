# MLflow cheatsheet

### How to create experiments and organize mlruns ?
`mlflow experiments create -n experiment_name`

### How to list experiments ?
`mlflow experiments list`

### How to explore mlruns ?
`mlflow ui -p port`

### How to run ?
`mlflow run path/to/project/directory --experiment-name=experiment_name -e entry_point -P parameter_name=parameter_value`

### How to serve predictions to a model (not served) ?
`mlflow models predict -m runs:/run_id/model_directory -i path/to/prediction.json -o path/to/output/file`

### How to serve a model ?
`mlflow models serve -m runs:/run_id/model_directory -p port`

### How to serve predictions to a model (served) ?
`curl -X POST -H 'Content-Type: application/json' -d @path/to/prediction_set.json http://127.0.0.1:port/invocations`

<[Précédent](entry_points.md) | [Suivant](diff_template.md)>
