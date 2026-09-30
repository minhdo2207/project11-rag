.PHONY: install format lint test check

install:
	pip install -r requirements.txt

format:
	ruff format .

lint:
	ruff check .

test:
	python -m pytest -q

# chay het truoc khi push
check: format lint test
