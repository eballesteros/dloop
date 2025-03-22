.PHONY: lint lint-fix format test

lint:
	poetry run ruff check .

lint-fix:
	poetry run ruff check --fix .

format:
	poetry run ruff format .

test:
	poetry run pytest

# Run both lint-fix and format
fix: lint-fix format