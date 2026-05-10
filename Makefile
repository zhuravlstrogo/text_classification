.PHONY: fmt lint test

fmt:
	python -m black .
	python -m isort .

lint:
	python -m pylint src scripts tests || true

test:
	python -m pytest -q

