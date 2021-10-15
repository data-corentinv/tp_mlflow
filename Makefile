SOURCE_DIR = foodcast

COVERAGE_OPTIONS = --cov-branch --cov-config coverage/.coveragerc --cov-report term --cov-report html

.PHONY: tests coverage notebooks

lint:
	flake8 $(SOURCE_DIR)

check:
	mypy . --strict --config-file .mypy.ini

tests:
	pytest -s tests/

coverage:
	py.test $(COVERAGE_OPTIONS) --cov=$(SOURCE_DIR) tests/ | tee coverage/coverage.txt
	mv .coverage coverage

notebooks:
	jupytext --to ipynb notebooks/*.md
	jupytext --to ipynb notebooks/.*.md

